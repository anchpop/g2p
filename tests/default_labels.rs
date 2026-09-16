//! Ten public outputs captured before the shared-type extraction: all non-Hindi
//! bytes and the Current Hindi row are unchanged. Hindi still defaults to the
//! trained Current labels; its selector is now private. The eleventh, historical
//! Legacy output lives in hindi-legacy.json and is checked by an internal Hindi
//! test; it is no longer a public option.
use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn supported_model_labels_remain_byte_identical() {
    let requests: Vec<_> = include_str!("fixtures/default-requests.jsonl")
        .lines()
        .collect();
    let responses: Vec<_> = include_str!("fixtures/default-responses.jsonl")
        .lines()
        .collect();
    assert_eq!(requests.len(), 10);
    assert_eq!(requests.len(), responses.len());
    let mut input = String::new();
    let mut expected = String::new();
    for (request, response) in requests.into_iter().zip(responses) {
        let request_value: serde_json::Value = serde_json::from_str(request).unwrap();
        if !cfg!(feature = "japanese") && request_value["lang"] == "jpn" {
            continue;
        }
        input.push_str(request);
        input.push('\n');
        expected.push_str(response);
        expected.push('\n');
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_g2p"))
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start g2p serve");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
}
