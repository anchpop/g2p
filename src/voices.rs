//! Private language/variety-to-engine mapping, shared by the engine and the
//! temporary legacy JSON adapter. Neither module is part of the public API.

use g2p_types::Variety;

pub(crate) const ESPEAK_VOICES: &[(&str, Variety, &str)] = &[
    ("eng", Variety::Default, "en-us"),
    ("deu", Variety::Default, "de"),
    ("fra", Variety::Default, "fr-fr"),
    ("ita", Variety::Default, "it"),
    ("por", Variety::Default, "pt-br"),
    ("spa", Variety::Default, "es"),
    ("rus", Variety::Default, "ru"),
    ("sqi", Variety::Default, "sq"),
    ("ara", Variety::Default, "ar"),
    ("hye", Variety::Default, "hy"),
    ("yue", Variety::Default, "yue"),
    ("hrv", Variety::Default, "hr"),
    ("ces", Variety::Default, "cs"),
    ("dan", Variety::Default, "da"),
    ("fas", Variety::Default, "fa"),
    ("nld", Variety::Default, "nl"),
    ("fin", Variety::Default, "fi"),
    ("hat", Variety::Default, "ht"),
    ("heb", Variety::Default, "he"),
    ("hun", Variety::Default, "hu"),
    ("isl", Variety::Default, "is"),
    ("ind", Variety::Default, "id"),
    ("gle", Variety::Default, "ga"),
    ("ell", Variety::Default, "el"),
    ("nor", Variety::Default, "nb"),
    ("pol", Variety::Default, "pl"),
    ("pan", Variety::Default, "pa"),
    ("ron", Variety::Default, "ro"),
    ("swa", Variety::Default, "sw"),
    ("swe", Variety::Default, "sv"),
    ("tur", Variety::Default, "tr"),
    ("ukr", Variety::Default, "uk"),
    ("urd", Variety::Default, "ur"),
    ("vie", Variety::Default, "vi"),
    ("spa", Variety::LatinAmerican, "es-419"),
    ("spa", Variety::European, "es"),
    // G2P can replay both Portuguese training varieties. Independently, Yap
    // excludes European Portuguese clips and offers no European accepted-reading
    // variant when scoring learners; that is product policy, not engine capability.
    ("por", Variety::Brazilian, "pt-br"),
    ("por", Variety::European, "pt"),
];
