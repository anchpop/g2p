//! Pronunciation choices shared by g2p and its consumers.

/// A supported language and pronunciation variety, selected together.
/// Backend routing belongs to g2p. Regional variants share a language code
/// for consumers such as text analysis that do not distinguish pronunciation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Language {
    #[serde(rename = "eng")]
    English,
    #[serde(rename = "fra")]
    French,
    #[serde(rename = "spa-ES")]
    SpanishEuro,
    #[serde(rename = "spa-419")]
    SpanishLatinAmerica,
    #[serde(rename = "por-BR")]
    PortugueseBrazil,
    #[serde(rename = "por-PT")]
    PortugueseEuro,
    #[serde(rename = "deu")]
    German,
    #[serde(rename = "ita")]
    Italian,
    #[serde(rename = "rus")]
    Russian,
    #[serde(rename = "jpn")]
    Japanese,
    #[serde(rename = "kor")]
    Korean,
    #[serde(rename = "hin")]
    Hindi,
    #[serde(rename = "tha")]
    Thai,
    #[serde(rename = "zho-hans")]
    ChineseSimplified,
    #[serde(rename = "sqi")]
    Albanian,
    #[serde(rename = "ara")]
    Arabic,
    #[serde(rename = "hye")]
    Armenian,
    #[serde(rename = "yue")]
    Cantonese,
    #[serde(rename = "hrv")]
    Croatian,
    #[serde(rename = "ces")]
    Czech,
    #[serde(rename = "dan")]
    Danish,
    #[serde(rename = "fas")]
    Persian,
    #[serde(rename = "nld")]
    Dutch,
    #[serde(rename = "fin")]
    Finnish,
    #[serde(rename = "hat")]
    HaitianCreole,
    #[serde(rename = "heb")]
    Hebrew,
    #[serde(rename = "hun")]
    Hungarian,
    #[serde(rename = "isl")]
    Icelandic,
    #[serde(rename = "ind")]
    Indonesian,
    #[serde(rename = "gle")]
    Irish,
    #[serde(rename = "ell")]
    Greek,
    #[serde(rename = "nor")]
    Norwegian,
    #[serde(rename = "pol")]
    Polish,
    #[serde(rename = "pan")]
    Punjabi,
    #[serde(rename = "ron")]
    Romanian,
    #[serde(rename = "swa")]
    Swahili,
    #[serde(rename = "swe")]
    Swedish,
    #[serde(rename = "tur")]
    Turkish,
    #[serde(rename = "ukr")]
    Ukrainian,
    #[serde(rename = "urd")]
    Urdu,
    #[serde(rename = "vie")]
    Vietnamese,
}

impl Language {
    /// ISO 639-3 language code, with `zho-hans` for Simplified Mandarin.
    pub const fn code(self) -> &'static str {
        match self {
            Self::English => "eng",
            Self::French => "fra",
            Self::SpanishEuro => "spa",
            Self::SpanishLatinAmerica => "spa",
            Self::PortugueseBrazil => "por",
            Self::PortugueseEuro => "por",
            Self::German => "deu",
            Self::Italian => "ita",
            Self::Russian => "rus",
            Self::Japanese => "jpn",
            Self::Korean => "kor",
            Self::Hindi => "hin",
            Self::Thai => "tha",
            Self::ChineseSimplified => "zho-hans",
            Self::Albanian => "sqi",
            Self::Arabic => "ara",
            Self::Armenian => "hye",
            Self::Cantonese => "yue",
            Self::Croatian => "hrv",
            Self::Czech => "ces",
            Self::Danish => "dan",
            Self::Persian => "fas",
            Self::Dutch => "nld",
            Self::Finnish => "fin",
            Self::HaitianCreole => "hat",
            Self::Hebrew => "heb",
            Self::Hungarian => "hun",
            Self::Icelandic => "isl",
            Self::Indonesian => "ind",
            Self::Irish => "gle",
            Self::Greek => "ell",
            Self::Norwegian => "nor",
            Self::Polish => "pol",
            Self::Punjabi => "pan",
            Self::Romanian => "ron",
            Self::Swahili => "swa",
            Self::Swedish => "swe",
            Self::Turkish => "tur",
            Self::Ukrainian => "ukr",
            Self::Urdu => "urd",
            Self::Vietnamese => "vie",
        }
    }

    /// Resolve a language-only input using established defaults:
    /// European Spanish and Brazilian Portuguese.
    pub fn from_code(code: &str) -> Option<Self> {
        Some(match code {
            "eng" => Self::English,
            "fra" => Self::French,
            "spa" => Self::SpanishEuro,
            "por" => Self::PortugueseBrazil,
            "deu" => Self::German,
            "ita" => Self::Italian,
            "rus" => Self::Russian,
            "jpn" => Self::Japanese,
            "kor" => Self::Korean,
            "hin" => Self::Hindi,
            "tha" => Self::Thai,
            "zho-hans" => Self::ChineseSimplified,
            "sqi" => Self::Albanian,
            "ara" => Self::Arabic,
            "hye" => Self::Armenian,
            "yue" => Self::Cantonese,
            "hrv" => Self::Croatian,
            "ces" => Self::Czech,
            "dan" => Self::Danish,
            "fas" => Self::Persian,
            "nld" => Self::Dutch,
            "fin" => Self::Finnish,
            "hat" => Self::HaitianCreole,
            "heb" => Self::Hebrew,
            "hun" => Self::Hungarian,
            "isl" => Self::Icelandic,
            "ind" => Self::Indonesian,
            "gle" => Self::Irish,
            "ell" => Self::Greek,
            "nor" => Self::Norwegian,
            "pol" => Self::Polish,
            "pan" => Self::Punjabi,
            "ron" => Self::Romanian,
            "swa" => Self::Swahili,
            "swe" => Self::Swedish,
            "tur" => Self::Turkish,
            "ukr" => Self::Ukrainian,
            "urd" => Self::Urdu,
            "vie" => Self::Vietnamese,
            _ => return None,
        })
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::English => "English",
            Self::French => "French",
            Self::SpanishEuro => "Spanish",
            Self::SpanishLatinAmerica => "Spanish",
            Self::PortugueseBrazil => "Portuguese",
            Self::PortugueseEuro => "Portuguese",
            Self::German => "German",
            Self::Italian => "Italian",
            Self::Russian => "Russian",
            Self::Japanese => "Japanese",
            Self::Korean => "Korean",
            Self::Hindi => "Hindi",
            Self::Thai => "Thai",
            Self::ChineseSimplified => "Chinese (Simplified)",
            Self::Albanian => "Albanian",
            Self::Arabic => "Arabic",
            Self::Armenian => "Armenian",
            Self::Cantonese => "Cantonese",
            Self::Croatian => "Croatian",
            Self::Czech => "Czech",
            Self::Danish => "Danish",
            Self::Persian => "Persian",
            Self::Dutch => "Dutch",
            Self::Finnish => "Finnish",
            Self::HaitianCreole => "Haitian Creole",
            Self::Hebrew => "Hebrew",
            Self::Hungarian => "Hungarian",
            Self::Icelandic => "Icelandic",
            Self::Indonesian => "Indonesian",
            Self::Irish => "Irish",
            Self::Greek => "Greek",
            Self::Norwegian => "Norwegian",
            Self::Polish => "Polish",
            Self::Punjabi => "Punjabi",
            Self::Romanian => "Romanian",
            Self::Swahili => "Swahili",
            Self::Swedish => "Swedish",
            Self::Turkish => "Turkish",
            Self::Ukrainian => "Ukrainian",
            Self::Urdu => "Urdu",
            Self::Vietnamese => "Vietnamese",
        })
    }
}
