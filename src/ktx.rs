use sib::render::{RenderError, RenderResult, texture};

pub fn decode_ktx1_rgba8(bytes: &[u8], label: &str) -> RenderResult<texture::ImageRgba8> {
    const IDENTIFIER: &[u8; 12] = b"\xABKTX 11\xBB\r\n\x1A\n";
    const GL_UNSIGNED_BYTE: u32 = 0x1401;
    const GL_RGBA: u32 = 0x1908;
    const GL_RGBA8: u32 = 0x8058;
    const GL_SRGB8_ALPHA8: u32 = 0x8C43;
    const HEADER_SIZE: usize = 64;

    if bytes.len() < HEADER_SIZE + 4 {
        return Err(RenderError::message(format!(
            "{label} KTX file is too small"
        )));
    }
    if bytes.get(..IDENTIFIER.len()) != Some(IDENTIFIER.as_slice()) {
        return Err(RenderError::message(format!(
            "{label} is not a KTX 1 texture"
        )));
    }

    let endianness = read_ktx_u32(bytes, 12, label, "endianness")?;
    let gl_type = read_ktx_u32(bytes, 16, label, "GL type")?;
    let gl_type_size = read_ktx_u32(bytes, 20, label, "GL type size")?;
    let gl_format = read_ktx_u32(bytes, 24, label, "GL format")?;
    let internal_format = read_ktx_u32(bytes, 28, label, "internal format")?;
    let base_format = read_ktx_u32(bytes, 32, label, "base format")?;
    let width = read_ktx_u32(bytes, 36, label, "width")?;
    let height = read_ktx_u32(bytes, 40, label, "height")?;
    let depth = read_ktx_u32(bytes, 44, label, "depth")?;
    let array_elements = read_ktx_u32(bytes, 48, label, "array elements")?;
    let faces = read_ktx_u32(bytes, 52, label, "faces")?;
    let mip_levels = read_ktx_u32(bytes, 56, label, "mip levels")?;
    let key_value_bytes = read_ktx_u32(bytes, 60, label, "key/value bytes")? as usize;

    if endianness != 0x0403_0201 {
        return Err(RenderError::message(format!(
            "{label} uses unsupported KTX endianness"
        )));
    }
    if gl_type != GL_UNSIGNED_BYTE
        || gl_type_size != 1
        || gl_format != GL_RGBA
        || base_format != GL_RGBA
        || !matches!(internal_format, GL_RGBA8 | GL_SRGB8_ALPHA8)
    {
        return Err(RenderError::message(format!(
            "{label} must be an uncompressed RGBA8 or sRGBA8 KTX 1 texture"
        )));
    }
    if width == 0
        || height == 0
        || depth != 0
        || array_elements != 0
        || faces != 1
        || mip_levels == 0
    {
        return Err(RenderError::message(format!(
            "{label} has an unsupported KTX 1 layout"
        )));
    }

    let image_size_offset = HEADER_SIZE
        .checked_add(key_value_bytes)
        .map(align_to_4)
        .ok_or_else(|| RenderError::message(format!("{label} KTX header size overflow")))?;
    let image_size = read_ktx_u32(bytes, image_size_offset, label, "image size")? as usize;
    let data_offset = image_size_offset
        .checked_add(4)
        .map(align_to_4)
        .ok_or_else(|| RenderError::message(format!("{label} KTX data offset overflow")))?;
    let expected_size = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| RenderError::message(format!("{label} KTX dimensions overflow")))?;
    if image_size < expected_size {
        return Err(RenderError::message(format!(
            "{label} KTX base mip has {image_size} bytes; expected {expected_size}"
        )));
    }
    let data_end = data_offset
        .checked_add(expected_size)
        .ok_or_else(|| RenderError::message(format!("{label} KTX data size overflow")))?;
    let rgba = bytes
        .get(data_offset..data_end)
        .ok_or_else(|| RenderError::message(format!("{label} KTX base mip is truncated")))?
        .to_vec();

    texture::ImageRgba8::new(width, height, rgba)
}

fn read_ktx_u32(bytes: &[u8], offset: usize, label: &str, field: &str) -> RenderResult<u32> {
    let raw = bytes
        .get(offset..offset.saturating_add(4))
        .and_then(|slice| <[u8; 4]>::try_from(slice).ok())
        .ok_or_else(|| RenderError::message(format!("{label} KTX {field} is truncated")))?;
    Ok(u32::from_le_bytes(raw))
}

const fn align_to_4(value: usize) -> usize {
    value.saturating_add(3) & !3
}
