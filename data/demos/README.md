# Demo bundles

Each folder is one investigation. In the UI upload: image -> `photo.jpg`, audio -> `speech.wav`,
documents -> every `.txt`, metadata -> paste `metadata.json`, claimed speaker -> `claimed_speaker.txt`.

| Case | What it shows | Expected verdict |
|---|---|---|
| case1_authentic | image-less bundle: audio + two agreeing text sources + matching EXIF | authentic |
| case2_manipulated | noise-patched image + editor tag in EXIF | manipulated |
| case3_coordinated_fake | every artifact is individually clean, but the sources disagree on place, date and speaker | coordinated_synthetic |
| case4_insufficient | a single unverified message | insufficient_evidence |

Run all four from the command line: `python -m evaluation.run_demos`
