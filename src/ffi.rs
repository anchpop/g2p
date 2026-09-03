//! Raw bindings to the parts of the espeak-ng C API this crate uses.
//! Declarations mirror `espeak-ng/src/include/espeak-ng/{speak_lib,espeak_ng}.h`.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_short, c_uint, c_void};

/// `espeak_ng_STATUS` / `espeak_ERROR`. Zero is success.
pub type Status = c_int;
pub const ENS_OK: Status = 0;

/// `espeak_ng_OUTPUT_MODE`
pub const ENOUTPUT_MODE_SYNCHRONOUS: c_int = 0x0001;

/// `espeak_POSITION_TYPE`
pub const POS_CHARACTER: c_int = 1;

/// `espeak_Synth` flags
pub const ESPEAK_CHARS_AUTO: c_uint = 0;
pub const ESPEAK_PHONEMES: c_uint = 0x100;
pub const ESPEAK_ENDPAUSE: c_uint = 0x1000;

/// `espeak_SetPhonemeTrace` modes
pub const ESPEAK_PHONEMES_SHOW: c_int = 0x01;
pub const ESPEAK_PHONEMES_IPA: c_int = 0x02;

#[repr(C)]
pub struct espeak_VOICE {
    pub name: *const c_char,
    pub languages: *const c_char,
    pub identifier: *const c_char,
    pub gender: u8,
    pub age: u8,
    pub variant: u8,
    pub xx1: u8,
    pub score: c_int,
    pub spare: *mut c_void,
}

/// Opaque; only ever passed through.
#[repr(C)]
pub struct espeak_EVENT {
    _private: [u8; 0],
}

pub type espeak_ng_ERROR_CONTEXT = *mut c_void;
pub type SynthCallback = unsafe extern "C" fn(*mut c_short, c_int, *mut espeak_EVENT) -> c_int;
pub type PhonemeCallback = unsafe extern "C" fn(*const c_char) -> c_int;

unsafe extern "C" {
    pub fn espeak_ng_InitializePath(path: *const c_char);
    pub fn espeak_ng_Initialize(context: *mut espeak_ng_ERROR_CONTEXT) -> Status;
    pub fn espeak_ng_InitializeOutput(
        output_mode: c_int,
        buffer_length: c_int,
        device: *const c_char,
    ) -> Status;
    pub fn espeak_ng_SetVoiceByName(name: *const c_char) -> Status;
    pub fn espeak_ng_SetVoiceByProperties(voice_selector: *mut espeak_VOICE) -> Status;
    pub fn espeak_ng_GetStatusCodeMessage(status: Status, buffer: *mut c_char, length: usize);
    pub fn espeak_SetSynthCallback(callback: Option<SynthCallback>);
    pub fn espeak_SetPhonemeCallback(callback: Option<PhonemeCallback>);
    pub fn espeak_SetPhonemeTrace(phonememode: c_int, stream: *mut libc::FILE);
    pub fn espeak_Synth(
        text: *const c_void,
        size: usize,
        position: c_uint,
        position_type: c_int,
        end_position: c_uint,
        flags: c_uint,
        unique_identifier: *mut c_uint,
        user_data: *mut c_void,
    ) -> Status;
}

/// Human-readable message for a non-zero status.
pub fn status_message(status: Status) -> String {
    let mut buf = [0u8; 256];
    // SAFETY: buffer and length describe a valid writable region.
    unsafe { espeak_ng_GetStatusCodeMessage(status, buf.as_mut_ptr().cast(), buf.len()) };
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}
