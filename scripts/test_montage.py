import mne

montage1005 = mne.channels.make_standard_montage("standard_1005")
pos1005 = montage1005.get_positions()["ch_pos"]

test_channels = ['FP1', 'FP2', 'F7', 'F8', 'AF1', 'AF2', 'FZ', 'F4', 'F3', 'FC6', 'FC5', 'FC2', 'FC1', 
                 'T8', 'T7', 'CZ', 'C3', 'C4', 'CP5', 'CP6', 'CP1', 'CP2', 'P3', 'P4', 'PZ', 'P8', 'P7', 
                 'PO2', 'PO1', 'O2', 'O1', 'AF7', 'AF8', 'F5', 'F6', 'FT7', 'FT8', 'FPZ', 'FC4', 'FC3', 
                 'C6', 'C5', 'F2', 'F1', 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4', 'P5', 'P6', 'C1', 'C2', 
                 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ', 'P2', 'P1', 'CPZ']

matched = 0
for ch in test_channels:
    variations = [
        ch, ch.upper(), ch.capitalize(),
        ch[:-1] + 'z' if ch.endswith('Z') else ch,
        ch[0] + ch[1:].lower(),
        ch[:2].capitalize() + ch[2:]
    ]
    found = False
    for v in variations:
        if v in pos1005:
            matched += 1
            found = True
            break
    if not found:
        print("Unmatched:", ch)

print(f"Matched {matched}/{len(test_channels)}!")
