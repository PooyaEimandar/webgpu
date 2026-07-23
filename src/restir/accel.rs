#[cfg(not(target_arch = "wasm32"))]
use bvh::bounding_hierarchy::BoundingHierarchy;
use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHShape,
    bvh::{Bvh, BvhNode},
};
use bytemuck::{Pod, Zeroable};
use nalgebra034::Point3;
use sib::render::{RenderError, RenderResult, glam};

use super::scene::GpuTriangle;

const BVH_ASSET_MAGIC: &[u8; 8] = b"RSTBVH01";
const BVH_ASSET_VERSION: u32 = 1;
const BVH_ASSET_HEADER_SIZE: usize = 32;
const LEAF_BIT: u32 = 1 << 31;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBvhNode {
    pub min_bounds: [f32; 3],
    pub first: u32,
    pub max_bounds: [f32; 3],
    pub second: u32,
}

impl GpuBvhNode {
    pub fn is_leaf(self) -> bool {
        self.first & LEAF_BIT != 0
    }

    pub fn triangle_index(self) -> Option<u32> {
        self.is_leaf().then_some(self.second)
    }

    pub fn children(self) -> Option<(u32, u32)> {
        (!self.is_leaf()).then_some((self.first, self.second))
    }

    fn leaf(min: glam::Vec3, max: glam::Vec3, triangle_index: u32) -> Self {
        Self {
            min_bounds: min.to_array(),
            first: LEAF_BIT,
            max_bounds: max.to_array(),
            second: triangle_index,
        }
    }

    fn internal(min: glam::Vec3, max: glam::Vec3, left: u32, right: u32) -> Self {
        Self {
            min_bounds: min.to_array(),
            first: left,
            max_bounds: max.to_array(),
            second: right,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TriangleShape {
    min: glam::Vec3,
    max: glam::Vec3,
    node_index: usize,
}

impl Bounded<f32, 3> for TriangleShape {
    fn aabb(&self) -> Aabb<f32, 3> {
        Aabb::with_bounds(
            Point3::new(self.min.x, self.min.y, self.min.z),
            Point3::new(self.max.x, self.max.y, self.max.z),
        )
    }
}

impl BHShape<f32, 3> for TriangleShape {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

pub fn build_gpu_bvh(triangles: &[GpuTriangle]) -> RenderResult<Vec<GpuBvhNode>> {
    if triangles.is_empty() {
        return Err(RenderError::message("cannot build a BVH without triangles"));
    }
    if triangles.len() > u32::MAX as usize {
        return Err(RenderError::message(
            "triangle count exceeds the GPU BVH index range",
        ));
    }

    let mut shapes = triangles
        .iter()
        .map(|triangle| {
            let (min, max) = triangle_bounds(triangle)?;
            Ok(TriangleShape {
                min,
                max,
                node_index: 0,
            })
        })
        .collect::<RenderResult<Vec<_>>>()?;

    #[cfg(not(target_arch = "wasm32"))]
    let bvh = Bvh::<f32, 3>::build_par(&mut shapes);
    #[cfg(target_arch = "wasm32")]
    let bvh = Bvh::<f32, 3>::build(&mut shapes);

    if bvh.nodes.len() >= LEAF_BIT as usize {
        return Err(RenderError::message(
            "BVH node count exceeds the compact GPU index range",
        ));
    }
    let nodes = bvh
        .nodes
        .iter()
        .map(|node| match *node {
            BvhNode::Node {
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
                ..
            } => {
                let aabb = child_l_aabb.join(&child_r_aabb);
                GpuBvhNode::internal(
                    glam::Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
                    glam::Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
                    child_l_index as u32,
                    child_r_index as u32,
                )
            }
            BvhNode::Leaf { shape_index, .. } => {
                let aabb = shapes[shape_index].aabb();
                GpuBvhNode::leaf(
                    glam::Vec3::new(aabb.min.x, aabb.min.y, aabb.min.z),
                    glam::Vec3::new(aabb.max.x, aabb.max.y, aabb.max.z),
                    shape_index as u32,
                )
            }
        })
        .collect::<Vec<_>>();

    validate_gpu_bvh(&nodes, triangles.len())?;
    Ok(nodes)
}

pub fn refit_gpu_bvh(nodes: &mut [GpuBvhNode], triangles: &[GpuTriangle]) -> RenderResult<()> {
    validate_gpu_bvh(nodes, triangles.len())?;
    let mut state = vec![0_u8; nodes.len()];
    let mut stack = Vec::with_capacity(128);
    stack.push((0_usize, false));

    while let Some((index, postorder)) = stack.pop() {
        let node = *nodes
            .get(index)
            .ok_or_else(|| RenderError::message("BVH refit reached an invalid node"))?;
        if postorder {
            let (left, right) = node
                .children()
                .ok_or_else(|| RenderError::message("BVH leaf entered the internal refit path"))?;
            let left = *nodes
                .get(left as usize)
                .ok_or_else(|| RenderError::message("BVH left child is out of range"))?;
            let right = *nodes
                .get(right as usize)
                .ok_or_else(|| RenderError::message("BVH right child is out of range"))?;
            let min = glam::Vec3::from_array(left.min_bounds)
                .min(glam::Vec3::from_array(right.min_bounds));
            let max = glam::Vec3::from_array(left.max_bounds)
                .max(glam::Vec3::from_array(right.max_bounds));
            nodes[index].min_bounds = min.to_array();
            nodes[index].max_bounds = max.to_array();
            state[index] = 2;
        } else if let Some(triangle_index) = node.triangle_index() {
            let triangle = triangles.get(triangle_index as usize).ok_or_else(|| {
                RenderError::message("BVH leaf references an unavailable triangle")
            })?;
            let (min, max) = triangle_bounds(triangle)?;
            nodes[index].min_bounds = min.to_array();
            nodes[index].max_bounds = max.to_array();
            state[index] = 2;
        } else {
            if state[index] != 0 {
                return Err(RenderError::message("BVH contains a cycle or shared child"));
            }
            state[index] = 1;
            let (left, right) = node
                .children()
                .ok_or_else(|| RenderError::message("BVH internal node has no children"))?;
            stack.push((index, true));
            stack.push((right as usize, false));
            stack.push((left as usize, false));
        }
    }

    if state.iter().any(|state| *state != 2) {
        return Err(RenderError::message(
            "BVH contains nodes that are not reachable from its root",
        ));
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn encode_gpu_bvh(triangles: &[GpuTriangle], nodes: &[GpuBvhNode]) -> RenderResult<Vec<u8>> {
    ensure_little_endian()?;
    validate_gpu_bvh(nodes, triangles.len())?;
    let triangle_count = u32::try_from(triangles.len())
        .map_err(|_| RenderError::message("triangle count exceeds the BVH asset format"))?;
    let node_count = u32::try_from(nodes.len())
        .map_err(|_| RenderError::message("node count exceeds the BVH asset format"))?;
    let payload = bytemuck::cast_slice(nodes);
    let mut bytes = Vec::with_capacity(BVH_ASSET_HEADER_SIZE + payload.len());
    bytes.extend_from_slice(BVH_ASSET_MAGIC);
    bytes.extend_from_slice(&BVH_ASSET_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(std::mem::size_of::<GpuBvhNode>() as u32).to_le_bytes());
    bytes.extend_from_slice(&triangle_count.to_le_bytes());
    bytes.extend_from_slice(&node_count.to_le_bytes());
    bytes.extend_from_slice(&geometry_fingerprint(triangles).to_le_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

pub fn decode_gpu_bvh(bytes: &[u8], triangles: &[GpuTriangle]) -> RenderResult<Vec<GpuBvhNode>> {
    ensure_little_endian()?;
    if bytes.get(..8) != Some(BVH_ASSET_MAGIC.as_slice()) {
        return Err(RenderError::message(
            "Sponza BVH asset has an invalid magic value",
        ));
    }
    let version = read_u32(bytes, 8)?;
    if version != BVH_ASSET_VERSION {
        return Err(RenderError::message(format!(
            "Sponza BVH asset version {version} is unsupported"
        )));
    }
    let node_stride = read_u32(bytes, 12)? as usize;
    if node_stride != std::mem::size_of::<GpuBvhNode>() {
        return Err(RenderError::message(
            "Sponza BVH asset node layout does not match this build",
        ));
    }
    let triangle_count = read_u32(bytes, 16)? as usize;
    if triangle_count != triangles.len() {
        return Err(RenderError::message(
            "Sponza BVH asset triangle count does not match the glTF geometry",
        ));
    }
    let node_count = read_u32(bytes, 20)? as usize;
    let fingerprint = read_u64(bytes, 24)?;
    if fingerprint != geometry_fingerprint(triangles) {
        return Err(RenderError::message(
            "Sponza BVH asset is stale for the current glTF geometry",
        ));
    }
    let payload_size = node_count
        .checked_mul(node_stride)
        .ok_or_else(|| RenderError::message("Sponza BVH asset size overflowed"))?;
    let payload = bytes
        .get(BVH_ASSET_HEADER_SIZE..)
        .filter(|payload| payload.len() == payload_size)
        .ok_or_else(|| RenderError::message("Sponza BVH asset payload is truncated"))?;
    let mut nodes = Vec::with_capacity(node_count);
    for chunk in payload.chunks_exact(node_stride) {
        nodes.push(bytemuck::pod_read_unaligned(chunk));
    }
    validate_gpu_bvh(&nodes, triangles.len())?;
    Ok(nodes)
}

fn validate_gpu_bvh(nodes: &[GpuBvhNode], triangle_count: usize) -> RenderResult<()> {
    if nodes.is_empty() {
        return Err(RenderError::message("GPU BVH contains no nodes"));
    }
    let mut leaf_count = 0_usize;
    for (index, node) in nodes.iter().copied().enumerate() {
        let min = glam::Vec3::from_array(node.min_bounds);
        let max = glam::Vec3::from_array(node.max_bounds);
        if !min.is_finite() || !max.is_finite() || !min.cmple(max).all() {
            return Err(RenderError::message("GPU BVH contains invalid bounds"));
        }
        if let Some(triangle_index) = node.triangle_index() {
            if triangle_index as usize >= triangle_count {
                return Err(RenderError::message(
                    "GPU BVH leaf triangle index is out of range",
                ));
            }
            leaf_count += 1;
        } else if let Some((left, right)) = node.children()
            && (left as usize >= nodes.len()
                || right as usize >= nodes.len()
                || left as usize == index
                || right as usize == index)
        {
            return Err(RenderError::message(
                "GPU BVH internal child index is invalid",
            ));
        }
    }
    if leaf_count != triangle_count {
        return Err(RenderError::message(
            "GPU BVH does not contain one leaf per triangle",
        ));
    }
    Ok(())
}

fn triangle_bounds(triangle: &GpuTriangle) -> RenderResult<(glam::Vec3, glam::Vec3)> {
    let p0 = triangle_position(triangle.p0, "first")?;
    let p1 = triangle_position(triangle.p1, "second")?;
    let p2 = triangle_position(triangle.p2, "third")?;
    let epsilon = glam::Vec3::splat(1.0e-5);
    Ok((p0.min(p1).min(p2) - epsilon, p0.max(p1).max(p2) + epsilon))
}

fn triangle_position(position: [f32; 4], label: &str) -> RenderResult<glam::Vec3> {
    let position = glam::Vec3::new(position[0], position[1], position[2]);
    if position.is_finite() {
        Ok(position)
    } else {
        Err(RenderError::message(format!(
            "invalid {label} triangle position"
        )))
    }
}

fn geometry_fingerprint(triangles: &[GpuTriangle]) -> u64 {
    bytemuck::cast_slice::<GpuTriangle, u8>(triangles)
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn read_u32(bytes: &[u8], offset: usize) -> RenderResult<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
        .ok_or_else(|| RenderError::message("Sponza BVH asset header is truncated"))?;
    Ok(u32::from_le_bytes(value))
}

fn read_u64(bytes: &[u8], offset: usize) -> RenderResult<u64> {
    let value = bytes
        .get(offset..offset + 8)
        .and_then(|bytes| <[u8; 8]>::try_from(bytes).ok())
        .ok_or_else(|| RenderError::message("Sponza BVH asset header is truncated"))?;
    Ok(u64::from_le_bytes(value))
}

fn ensure_little_endian() -> RenderResult<()> {
    if cfg!(target_endian = "little") {
        Ok(())
    } else {
        Err(RenderError::message(
            "the compact BVH asset currently supports little-endian targets",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle(offset: glam::Vec3) -> GpuTriangle {
        GpuTriangle::from_positions(offset, offset + glam::Vec3::X, offset + glam::Vec3::Y, 0)
    }

    #[test]
    fn flattened_bvh_contains_one_leaf_per_triangle() -> RenderResult<()> {
        let triangles = [triangle(glam::Vec3::ZERO)];
        let nodes = build_gpu_bvh(&triangles)?;
        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_leaf());
        assert_eq!(nodes[0].triangle_index(), Some(0));
        Ok(())
    }

    #[test]
    fn compact_asset_round_trips() -> RenderResult<()> {
        let triangles = [triangle(glam::Vec3::ZERO), triangle(glam::Vec3::splat(2.0))];
        let nodes = build_gpu_bvh(&triangles)?;
        let encoded = encode_gpu_bvh(&triangles, &nodes)?;
        let decoded = decode_gpu_bvh(&encoded, &triangles)?;
        assert_eq!(
            bytemuck::cast_slice::<GpuBvhNode, u8>(&decoded),
            bytemuck::cast_slice::<GpuBvhNode, u8>(&nodes)
        );
        Ok(())
    }

    #[test]
    fn refit_updates_root_bounds_without_rebuilding_topology() -> RenderResult<()> {
        let original = [triangle(glam::Vec3::ZERO), triangle(glam::Vec3::splat(2.0))];
        let moved = [
            triangle(glam::Vec3::splat(4.0)),
            triangle(glam::Vec3::splat(6.0)),
        ];
        let mut nodes = build_gpu_bvh(&original)?;
        let topology = nodes
            .iter()
            .map(|node| (node.first, node.second))
            .collect::<Vec<_>>();
        refit_gpu_bvh(&mut nodes, &moved)?;
        assert_eq!(
            topology,
            nodes
                .iter()
                .map(|node| (node.first, node.second))
                .collect::<Vec<_>>()
        );
        assert!(nodes[0].min_bounds[0] > 3.9);
        assert!(nodes[0].max_bounds[0] > 6.9);
        Ok(())
    }
}
