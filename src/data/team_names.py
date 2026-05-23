"""Canonical team name mapping — resolves all spelling variants to one standard name."""

ALIASES: dict[str, str] = {
    # USA
    "United States": "USA",
    "United States of America": "USA",
    "US": "USA",

    # Korea
    "South Korea": "Korea Republic",
    "Korea": "Korea Republic",
    "Korea DPR": "North Korea",

    # Ivory Coast
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",

    # China
    "China": "China PR",

    # Bosnia
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Bosnia & Herzegovina": "Bosnia-Herzegovina",

    # Macedonia
    "North Macedonia": "Macedonia",
    "FYR Macedonia": "Macedonia",

    # Congo
    "DR Congo": "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Republic of Congo": "Congo",

    # Iran
    "Iran": "IR Iran",

    # Kyrgyzstan
    "Kyrgyz Republic": "Kyrgyzstan",

    # Misc common variants
    "Czech Republic": "Czechia",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Eswatini": "Swaziland",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Timor-Leste": "East Timor",
}


def canonical(name: str) -> str:
    """Return the canonical team name for a given input string."""
    return ALIASES.get(name, name)
