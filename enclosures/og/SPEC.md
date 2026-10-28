# Oi Speaker enclosure: parts and key specs

Reference sheet for the `og` design. The numbers come from `oi_og_enclosure.py`; regenerate `out/cutlist.csv` and
`out/report.txt` for the current values after changing any parameter.

## Overall
| | |
|---|---|
| Body (W × D × H) | 150 × 210 × 242 mm |
| Height with mic cap | 272 mm |
| Stock | 18 mm ply for the body, 9 mm ply for the cap and tray. Measure the real thickness and set `T` / `T9` |
| Joints | 6 mm dados and rabbets in the 18 mm stock, 4.5 mm rabbets in the 9 mm stock, 6 mm router bit |

## Acoustic chamber
| | |
|---|---|
| Inside (W × D × H) | 114 × 174 × 120 mm |
| Volume | 2.380 L gross, **2.207 L net** (after the driver 0.12, PR 0.04 and conduit 0.014 L) |
| Tuning | 69.5 Hz with no added mass on the PR, confirmed in the 00 Audio simulator (`project.00sim`) |
| Driver | Dayton **RS100-8**: 98 mm OD, 77.7 mm cutout, 5 mm flange rebate. Centred at x=75, 164 mm off the floor |
| Passive radiator | Dayton **ND90-PR**: 103.4 mm OD, 85.6 mm cutout, 104.4 mm × 5 mm recess, 60.7 mm deep. Fs 53.8 Hz, Qms 6.15, Mms 8.1 g, Vas 1.5 L, Sd 31.2 cm², Xmax 9 mm. Centred on the back panel, x=75 |
| Driver to PR gap inside | 72.3 mm |
| Cable conduit | 12 mm OD tube (~9 mm bore, takes a USB-C plug) against the left wall, 38 mm in from the back (x=24, y=172), so it comes up into the cap clear of the corner pad. Sealed at both ends. Any wider and the PR has to move off centre |
| Speaker wire hole | 8 mm through the shelf, rear-right corner |

### PR tuning
```
Fb = 1/(2π) × √( (1/Cms + 1/Cab) / (Mms + added) )
Cab = Vb / (ρc² × Sd²)        ρc² ≈ 1.42 × 10⁵ Pa
```
In 2.2 L the springs come to 917 N/m (suspension) + 628 N/m (air) = 1545 N/m, so the total mass for a tuning is
`1545 / (2π × Fb)²`, and the added mass is that minus 8.1 g.

| Added mass | Fb | Notch (Fp) |
|---|---|---|
| none | 69.5 Hz | 53.8 Hz |
| one disc, 7.3 g | 50.4 Hz | 38.8 Hz |
| both discs, 14.6 g | 41.5 Hz | 32.0 Hz |

The chosen tuning is no added mass, with the wing nut left off: the response is flattest that way. Check the PR's
excursion stays under 9 mm at full power.

## Cut list
| Part | Qty | Thickness | Size (mm) | Notes |
|---|---|---|---|---|
| side_L, side_R | 2 | 18 | 210 × 242 | Face A inside |
| bottom | 1 | 18 | 126 × 210 | |
| top | 1 | 18 | 126 × 210 | Face A underside. Dowel holes for the cap |
| shelf | 1 | 18 | 126 × 210 | Seals the chamber; pockets on both faces |
| baffle | 1 | 18 | 126 × 132 | Driver rebate on the front |
| back | 1 | 18 | 126 × 132 | PR rebate on the outside |
| fin_h | 1 | 18 | 126 × 30 | Front cross, horizontal |
| fin_v | 1 | 18 | 30 × 80 | Front cross, vertical (halving joint) |
| cap_front_back | 2 | 9 | 150 × 25.5 | Arched bridge walls, front and back are the same part |
| cap_left_right | 2 | 9 | 201 × 25.5 | Arched bridge walls, left and right are the same part |
| cap_grille | 1 | 9 | 150 × 210 | Slotted top, rabbeted for the walls |
| cap_pad_fl_fr_rl_rr | 4 | 18 | 20 × 20 | Offcuts. Dowel on the diagonal, so any pad fits any corner, any way round |
| tray_plate | 1 | 9 | 113 × 67 | Back connector panel |
| tray_floor | 1 | 9 | 113 × 174.5 | The Pi sits on it. 113 × 169.5 is usable |

## Mic cap
- The legs sit flush on the body. The arched gaps are 9 mm tall (1 mm under the mic board, so its edge is hidden), 90 mm wide front and back, and 150 mm wide on
  the sides.
- It's located by **4 × 6 mm hardwood dowels**, 13 mm long, glued into the pads and dropping loose into holes in
  the top. It's held on by its own weight.
- The XVF3800 mic array stands 10 mm above the body top on its standoffs. Its USB-C and jack stand 8 mm overall, leaving
  3 mm under the grille, which is 21 mm above the body top. The XIAO underneath is 5.5 mm clear of the top.

## Pi bay and tray
| | |
|---|---|
| Bay inside | 114 × 180 (behind the fins) × 68 mm |
| Electronics | Raspberry Pi 5 + **HiFiBerry Amp4** on 8 mm M2.5 standoffs. The stack tops out 64 mm off the floor, 22 mm under the shelf |
| Orientation | USB/Ethernet end faces the back (`PI_ROT=90`). The amp's DC jack and speaker terminals face the right wall |
| Pi plugs | 2 × USB-A in the lower ports (mic, external), modelled at 16 × 9 × 38 mm |
| Tray | Slides in from the back and stops on the fins |

Gaps from the Pi's plugs: barrel jack 46.5 mm, panel USB moulding 13 mm, button crimps 8 mm, right wall 9 mm (DC
plug), left wall 5 mm. Bend the button's crimp leads upward when wiring.

## Back panel (`tray_plate`), seen from behind
```
---------------------------- [finger notch]
 | | | | | | |      7 slots, 6.5 wide, 11.5 pitch
[jack]     [usb]     [button]   row 43 mm off the floor
```
The jack is on the Amp4's side, so its lead runs straight down that wall to the amp.

| Part | Product | Hole | Key dimensions |
|---|---|---|---|
| DC jack | Adafruit **5607** panel-mount 5.5/2.1 lead, 30 cm | 22.5 mm | M22 × 14 mm thread, 28 mm × 2.5 mm flange, 30 mm nut, 33.5 mm overall (22 mm behind the panel) |
| USB | Adafruit panel-mount USB-A extension, 16 cm | 15 × 8 mm slot | Ears at 29 mm pitch, screwed from inside with No.4 × 6 mm wood screws (2 mm pilots, ~5 mm deep). Nothing shows outside |
| Button | Adafruit **1505** 16 mm momentary, black | 16.2 mm | 18 mm bezel, 15.6 mm × 18.5 mm thread, 29.5 mm overall, 2.8 mm quick-connect tags |

## Fixings
- **Driver and PR:** black M4 button-head hex screws (ISO 7380) into threaded inserts. Check the flange holes
  first; drill on assembly.
- **USB:** 2 × No.4 × 6 mm pan-head wood screws from inside, through the ears (drill the ears out to 3.2 mm if
  they're threaded). Not the supplied M3 × 12s: they'd come out the front.
- **Cap:** 6 mm hardwood dowels.
- **Pi:** M2.5 × 8 mm standoffs.
- **XVF3800:** M2 × 10 mm standoffs, into 1.6 mm pilots in the top.

## To measure when the parts arrive
| Parameter | What to measure |
|---|---|
| `T`, `T9` | Real ply thickness |
| `PR_FLANGE`, `PR_FLATS` | Flange thickness and shape on the ND90-PR |
| `DRV_FLANGE` | Flange thickness on the RS100-8 |
| `USB_CUT` | USB stub body |
| `BARREL_NUT_T` | Barrel jack nut thickness; 5 mm or less fits |
| `USB_PLUG`, `DC_PLUG`, `BARREL_BOOT` | Plug and boot sizes |
| `BUTTON_CRIMPS` | Button crimp length; currently 8 mm clear of the Pi's plugs |

Driver and PR screw positions aren't in the DXFs yet, and nor are the XVF3800's mounting holes; both are drilled
on assembly.
