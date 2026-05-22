# Qualitative Error Analysis Preview

Split: `test` | Threshold: `0.66`

## Blocked false positives

### Example 1
- Pair ID: `6175893#51848095`
- Label: `0`
- Neural score: `0.996576513311884`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["model_token_conflict"]`
- Auto-category: different model identifiers
- Explanation: Neural-only would merge this non-match (score 0.997); governance blocked due to: different model identifiers. Violated: model_token_conflict.
- Left title: Đồng hồ Tissot T006.207.11.038.00
- Right title: Tissot T109.410.11.033.00
- Left brand: nan
- Right brand: nan

### Example 2
- Pair ID: `6175893#26986067`
- Label: `0`
- Neural score: `0.9944621164339682`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["model_token_conflict"]`
- Auto-category: different model identifiers
- Explanation: Neural-only would merge this non-match (score 0.994); governance blocked due to: different model identifiers. Violated: model_token_conflict.
- Left title: Đồng hồ Tissot T006.207.11.038.00
- Right title: Tissot Mens T-Sport V8 Swissmatic White Dial Bracelet Watch T106.407.11.031.01
- Left brand: nan
- Right brand: Tissot

### Example 3
- Pair ID: `66481726#35452020`
- Label: `0`
- Neural score: `0.992156025399696`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["variant_modifier_conflict"]`
- Auto-category: variant mismatch
- Explanation: Neural-only would merge this non-match (score 0.992); governance blocked due to: variant mismatch. Violated: variant_modifier_conflict.
- Left title: iPhone SE Screen - Fix Kit / Black / New / Fix Kit / White / New / Part Only / Black / New / Part Only / White / New
- Right title: iPhone 7 Plus Lightning Connector Assembly - Gray / New / Fix Kit / White / New / Fix Kit / Black / New / Fix Kit / Gray / New / Part Only / White / New / Part Only / Black / New / Part Only
- Left brand: nan
- Right brand: nan

### Example 4
- Pair ID: `6175893#55154351`
- Label: `0`
- Neural score: `0.9900528632829604`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["model_token_conflict"]`
- Auto-category: different model identifiers
- Explanation: Neural-only would merge this non-match (score 0.990); governance blocked due to: different model identifiers. Violated: model_token_conflict.
- Left title: Đồng hồ Tissot T006.207.11.038.00
- Right title: TISSOT EVERYTIME MEDIUM T109.410.11.033.00 (T109.410.11.033.00)
- Left brand: nan
- Right brand: nan

### Example 5
- Pair ID: `6175893#64388647`
- Label: `0`
- Neural score: `0.9892335209744726`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["model_token_conflict"]`
- Auto-category: different model identifiers
- Explanation: Neural-only would merge this non-match (score 0.989); governance blocked due to: different model identifiers. Violated: model_token_conflict.
- Left title: Đồng hồ Tissot T006.207.11.038.00
- Right title: TISSOT T095.417.11.067.00 QUICKSTER
- Left brand: nan
- Right brand: nan

## Blocked true positives

### Example 1
- Pair ID: `12722049#13640797`
- Label: `1`
- Neural score: `0.9798335314978656`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["variant_modifier_conflict"]`
- Auto-category: variant mismatch
- Explanation: True match rejected by governance (score 0.980); likely: variant mismatch. Violated: variant_modifier_conflict.
- Left title: Autel Robotics EVO II 6K Gimbal Camera
- Right title: Autel Robotics EVO II Pro Gimbal Camera 1\" 6K Camera for EVO 2 Drones, In Stock
- Left brand: nan
- Right brand: nan

### Example 2
- Pair ID: `29938853#87564521`
- Label: `1`
- Neural score: `0.975105505691407`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["model_token_conflict"]`
- Auto-category: different model identifiers
- Explanation: True match rejected by governance (score 0.975); likely: different model identifiers. Violated: model_token_conflict.
- Left title: Canon PGI-72 Ink Cartridge (Photo Magenta)CO90223
- Right title: Genuine Canon PGI-72 Photo Magenta Ink Cartridge (PGI72PMOEM)
- Left brand: Canon
- Right brand: Canon

### Example 3
- Pair ID: `7987227#31807881`
- Label: `1`
- Neural score: `0.9713324375048514`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["variant_modifier_conflict"]`
- Auto-category: variant mismatch
- Explanation: True match rejected by governance (score 0.971); likely: variant mismatch. Violated: variant_modifier_conflict.
- Left title: Switch CISCO WS-C2960XR-24PS-I
- Right title: Cisco Catalyst 2960-XR | WS-C2960XR-24PS-I Catalyst 2960XR 24 port 10/100/1000 PoE+ 370W, 4 x 1G SFP, IP Lite
- Left brand: Cisco
- Right brand: Cisco

### Example 4
- Pair ID: `18140756#51109166`
- Label: `1`
- Neural score: `0.9549614259729504`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["model_token_conflict"]`
- Auto-category: different model identifiers
- Explanation: True match rejected by governance (score 0.955); likely: different model identifiers. Violated: model_token_conflict.
- Left title: Ssd patriot burst 480gb 2.5 sata3 r/w speed: 560ms/s/540 mb/s
- Right title: Patriot BURST 480GB 2.5 SSD
- Left brand: Patriot
- Right brand: nan

### Example 5
- Pair ID: `995548#36270221`
- Label: `1`
- Neural score: `0.9516952709198978`
- Neural pred: `1`
- Governed pred: `0`
- Symbolic status: `invalid`
- Violated constraints: `["model_token_conflict"]`
- Auto-category: different model identifiers
- Explanation: True match rejected by governance (score 0.952); likely: different model identifiers. Violated: model_token_conflict.
- Left title: Seiko Prospex Automatic Divers 200M
- Right title: Seiko Mens Prospex Divers Automatic Blue Rubber Strap Watch SPB053J1
- Left brand: nan
- Right brand: Seiko

## Remaining false positives

### Example 1
- Pair ID: `56698149#45220776`
- Label: `0`
- Neural score: `0.9924204348548196`
- Neural pred: `1`
- Governed pred: `1`
- Symbolic status: `uncertain`
- Violated constraints: `[]`
- Auto-category: high lexical overlap without symbolic contradiction
- Explanation: Governed false positive (score 0.992); high lexical overlap without symbolic contradiction. Violated: none.
- Left title: Epson Ultra Glossy Photo Paper
- Right title: EPSON S041944 ultra glossy photo paper 300g/m², 13x18cm/50vel
- Left brand: Epson
- Right brand: nan

### Example 2
- Pair ID: `56159566#29009865`
- Label: `0`
- Neural score: `0.9889600346120624`
- Neural pred: `1`
- Governed pred: `1`
- Symbolic status: `valid`
- Violated constraints: `[]`
- Auto-category: missing symbolic evidence
- Explanation: Governed false positive (score 0.989); missing symbolic evidence. Violated: none.
- Left title: Tissot Mens T-Race Cycling Dark Blue Watch T111.417.37.441.06
- Right title: Tissot T111.417.37.441.05 T-Race Cycling Chronograph Black PVD 44mm
- Left brand: Tissot
- Right brand: nan

### Example 3
- Pair ID: `24931990#29009865`
- Label: `0`
- Neural score: `0.9880911398768218`
- Neural pred: `1`
- Governed pred: `1`
- Symbolic status: `valid`
- Violated constraints: `[]`
- Auto-category: high lexical overlap without symbolic contradiction
- Explanation: Governed false positive (score 0.988); high lexical overlap without symbolic contradiction. Violated: none.
- Left title: Tissot T-Race Chronograph Cycling Chronograph
- Right title: Tissot T111.417.37.441.05 T-Race Cycling Chronograph Black PVD 44mm
- Left brand: nan
- Right brand: nan

### Example 4
- Pair ID: `4395083#68021989`
- Label: `0`
- Neural score: `0.9874603493453024`
- Neural pred: `1`
- Governed pred: `1`
- Symbolic status: `uncertain`
- Violated constraints: `[]`
- Auto-category: missing symbolic evidence
- Explanation: Governed false positive (score 0.987); missing symbolic evidence. Violated: none.
- Left title: DJI Mavic 2 Enterprise Dual
- Right title: DJI Mavic 2 Enterprise (Zoom) w/ Smart Controller and DJI Enterprise Shield Basic
- Left brand: nan
- Right brand: nan

### Example 5
- Pair ID: `29009865#7185118`
- Label: `0`
- Neural score: `0.9867871546181628`
- Neural pred: `1`
- Governed pred: `1`
- Symbolic status: `valid`
- Violated constraints: `[]`
- Auto-category: missing symbolic evidence
- Explanation: Governed false positive (score 0.987); missing symbolic evidence. Violated: none.
- Left title: Tissot T111.417.37.441.05 T-Race Cycling Chronograph Black PVD 44mm
- Right title: Tissot T-Race Cycling T111.417.27.441.00
- Left brand: nan
- Right brand: nan

## False negatives

### Example 1
- Pair ID: `23833261#55728462`
- Label: `1`
- Neural score: `0.6577339363103516`
- Neural pred: `0`
- Governed pred: `0`
- Symbolic status: `valid`
- Violated constraints: `[]`
- FN type: `score_false_negative`
- Auto-category: low lexical overlap / paraphrase
- Explanation: True match missed: score 0.658 below threshold; low lexical overlap / paraphrase.
- Left title: Monitor 23.8\" ACER B247Ybmiprx, 16:9, IPS, LED, FHD 1920*1080, 4 ms, 250 cd/mp, 1000:1/ 100M:1, 178/178, pivot, VGA, HDMI, DP(1.2), UM.QB7EE.001
- Right title: Acer B7 B247Y bmiprx 23.8\" Full HD LED Mat Flat Zwart computer monitor
- Left brand: ACER
- Right brand: nan

### Example 2
- Pair ID: `10762448#31526766`
- Label: `1`
- Neural score: `0.655467647335566`
- Neural pred: `0`
- Governed pred: `0`
- Symbolic status: `valid`
- Violated constraints: `[]`
- FN type: `score_false_negative`
- Auto-category: missing brand evidence
- Explanation: True match missed: score 0.655 below threshold; missing brand evidence.
- Left title: APC Smart-UPS XL 48V Battery Pack - Battery enclosure - 48 V - 2 x Lead Acid- 5U
- Right title: SUA48XLBPAPC Smart-UPS XL 48V Battery Pack Tower/Rack Convertible
- Left brand: nan
- Right brand: nan

### Example 3
- Pair ID: `79368706#14576013`
- Label: `1`
- Neural score: `0.6525680978579173`
- Neural pred: `0`
- Governed pred: `0`
- Symbolic status: `valid`
- Violated constraints: `[]`
- FN type: `score_false_negative`
- Auto-category: missing brand evidence
- Explanation: True match missed: score 0.653 below threshold; missing brand evidence.
- Left title: Hotpoint BI WMHG 71284 UK Integrated Washing Machine - White
- Right title: Hotpoint BIWMHG71284 7kg 1200 Spin Integrated Washing Machine
- Left brand: nan
- Right brand: Hotpoint

### Example 4
- Pair ID: `9932421#8035591`
- Label: `1`
- Neural score: `0.6519279872786077`
- Neural pred: `0`
- Governed pred: `0`
- Symbolic status: `valid`
- Violated constraints: `[]`
- FN type: `score_false_negative`
- Auto-category: low lexical overlap / paraphrase
- Explanation: True match missed: score 0.652 below threshold; low lexical overlap / paraphrase.
- Left title: Fenix 21700 5000 mAh Li-ion Protected Battery
- Right title: FENIX ARE-A4 Multifunctional Smart Battery Charger for Popular Rechargeable Batteries
- Left brand: Fenix
- Right brand: FENIX

### Example 5
- Pair ID: `75922212#43606122`
- Label: `1`
- Neural score: `0.6518666514645616`
- Neural pred: `0`
- Governed pred: `0`
- Symbolic status: `valid`
- Violated constraints: `[]`
- FN type: `score_false_negative`
- Auto-category: low lexical overlap / paraphrase
- Explanation: True match missed: score 0.652 below threshold; low lexical overlap / paraphrase.
- Left title: A-DATA UV128 128GB USB3.0 Stick Black
- Right title: ADATA DashDrive UV128 128GB 128GB USB 3.0 BlackBlue USB flash drive
- Left brand: A-Data
- Right brand: nan
