//! Regression fixtures for the unified phonemization output.

#[test]
fn supported_model_labels_match_fixtures() {
    let requests: Vec<_> = include_str!("fixtures/default-requests.jsonl")
        .lines()
        .collect();
    let responses: Vec<_> = include_str!("fixtures/default-responses.jsonl")
        .lines()
        .collect();
    assert_eq!(requests.len(), 10);
    assert_eq!(requests.len(), responses.len());
    for (request, response) in requests.into_iter().zip(responses) {
        let request: serde_json::Value = serde_json::from_str(request).unwrap();
        if !cfg!(feature = "japanese") && request["lang"] == "jpn" {
            continue;
        }
        let language = serde_json::from_value(request["lang"].clone()).unwrap();
        let output = g2p::phonemize(language, request["text"].as_str().unwrap()).unwrap();
        let mut actual = serde_json::to_value(&output).unwrap();
        // The corpus fixtures store stress as numeric training labels.
        actual["stress"] =
            serde_json::to_value(output.stress.iter().map(|s| s.code()).collect::<Vec<_>>())
                .unwrap();
        let expected: serde_json::Value = serde_json::from_str(response).unwrap();
        assert_eq!(actual, expected, "{}", request["lang"]);
    }
}
