# Metabolic model component — central carbon metabolism

Description
- Documentation for a model component implementing parts of central carbon metabolism.
- Includes model construction notes, reaction stoichiometry, and thermodynamic handling (standard ΔG/ΔG°′ values).

Sources and citation
- Model structure and parameter choices: https://www.nature.com/articles/s41467-020-16549-2
- Standard Gibbs free energies (ΔG/ΔG°′) curated from Lehninger Principles of Biochemistry (Nelson & Cox). Cite these sources when reusing results.

Notes and assumptions
- Thermodynamic values are reported under the standard conditions used in the sources. Users must adjust ΔG for experimental conditions (pH, temperature, ionic strength) before quantitative comparisons.
- Reaction directions and enzyme assignments reflect literature-derived conventions; verify when integrating with other datasets.
- Maintain original citations when redistributing model-derived results.

Usage
- Consult function- and class-level documentation for parameter names, units, and expected inputs.
- Validate numeric units, reaction stoichiometry, and directionality when combining with other models.
- Recommended checks: mass balance, charge balance, and sanity checks of computed ΔG and equilibrium constants against reference values.

Contact / maintainers
- See repository metadata for author and maintainer details.
- Open issues or pull requests for updates, corrections, or source additions.
