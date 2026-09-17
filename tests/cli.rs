//! The command line accepts a combined language, never a positional engine voice.
use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_g2p"))
        .args(args)
        .output()
        .unwrap()
}

fn phonemes(args: &[&str]) -> serde_json::Value {
    let output = run(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(result.get("error").is_none(), "{result}");
    result["phonemes"].clone()
}

#[test]
fn cli_selects_spanish_and_portuguese_varieties() {
    let european = phonemes(&["--lang", "spa-ES", "cinco"]);
    let latin = phonemes(&["--lang", "spa-419", "cinco"]);
    assert_eq!(european[0], "θ");
    assert_eq!(latin[0], "s");
    let brazilian = phonemes(&["--lang", "por-BR", "dia", "noite"]);
    let european = phonemes(&["--lang", "por-PT", "dia", "noite"]);
    assert_ne!(brazilian, european);
}

#[test]
fn positional_raw_voices_and_malformed_options_are_rejected() {
    for args in [
        vec!["fr-fr", "bonjour"],
        vec!["--lang"],
        vec!["--lang", "spa"],
        vec!["--lang", "spa", "--variety"],
        vec!["--lang", "spa", "--variety", "european"],
        vec!["--lang", "spa", "--variety", "European", "cinco"],
        vec!["--lang", "spa", "--voice", "es", "cinco"],
    ] {
        let result = run(&args);
        assert_eq!(result.status.code(), Some(2), "{args:?}");
        assert!(result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
    }
}

#[test]
fn text_is_an_utterance_and_double_dash_ends_options() {
    assert_eq!(
        phonemes(&["--lang", "fra", "- Bonjour"]),
        phonemes(&["--lang", "fra", "Bonjour"]),
    );
    assert_eq!(
        phonemes(&["--lang", "fra", "--", "--bonjour"]),
        phonemes(&["--lang", "fra", "bonjour"]),
    );
}
