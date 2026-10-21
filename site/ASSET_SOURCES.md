# Asset sources

## Research content and recordings

Paper: **Detector-Guided Vocoder Training for Adversarial Text-to-Speech Synthesis**, Jiayan Cui, Davide Salvi, Paolo Bestagini and Stefano Tubaro, Politecnico di Milano. Method descriptions were checked against the manuscript supplied by the author. The paper is accepted at IEEE WIFS 2026; the title and citation follow the official polimi-ispl repository.

The 60 WAV files come from the author's existing [Audio Samples collection](https://h0yjmbl9to9.feishu.cn/docx/FH7mdd1dco6xDkxFlhPchve8nyh), accessed on 2026-10-19. There are six speaker references and 54 synthesized samples: six speakers × three prompts × three vocoders. Column mapping follows the source collection: reference, V, VR, VRT. The source filenames and SHA-256 values are preserved in `samples.json`, served beneath the `https://polimi-ispl.github.io/adversarial-tts/` base path.

The source identifies Gorou and Yae Miko as characters from **Genshin Impact** by **miHoYo**. These examples and references are presented with that attribution. The font license does not extend to the audio recordings or other research materials; no new blanket license for those materials is asserted here.

Prompt excerpts are reconstructed from attachment filenames and are not verified full transcripts. Speaker references have different text from the synthesized prompts. The source introduction has inconsistent row numbering; the site uses the explicit speaker labels in the actual table.

## Typography

Manrope, weights 400, 600, 700 and 800, retrieved from Google Fonts. The font is distributed under the SIL Open Font License 1.1; the full notice is in `site/assets/fonts/OFL.txt` (served as `assets/fonts/OFL.txt`).

- [Google Fonts Manrope source and license](https://github.com/google/fonts/tree/main/ofl/manrope)

## Artwork

The live hero and player waveforms use the stored peaks from `site/samples.json`. The hero shows Woman 29, “The wingtip”, for V, VR, and VRT. The shared SVG cover at `docs/assets/cover.svg` and `site/assets/cover.svg` uses the same three sets of 96 original peaks. Its bars use the same amplitude mapping as the player, with a fixed display height and width; they are visual summaries, not measurements of model quality. The two cover copies are byte-identical. The icon takes seven peak maxima from the same VRT sample.

The SVGs are local vector assets with text and shapes, without external rendering services, scripts, embedded raster imagery or new font dependencies. No stock imagery or newly synthesized speech is used.

## Deployment

The listening page is configured for `https://polimi-ispl.github.io/adversarial-tts/` from `site/` in the `polimi-ispl/adversarial-tts` repository using GitHub Actions. Only the static page, fonts, artwork, sample metadata and the existing public audio recordings are included. Model checkpoints and matching inference configurations are available separately at https://huggingface.co/Cocii/adversarial-tts. Model files are not included in the website repository. The older audio page remains available.
