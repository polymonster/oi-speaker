import base64, json
d = json.load(open('out/parts.json')); P = d['params']; D = d['derived']
specs = [
 ['Body W×D×H', f"{P['W']:.0f}×{D['DEPTH']:.1f}×{D['H']:.1f}"],
 ['With mic cap', f"{D['TOTAL_H']:.1f} mm"],
 ['Chamber (net)', f"{P['CHAMBER_NET_L']} L"],
 ['Chamber inside', f"{D['CH_W']:.0f}×{D['CH_D']:.1f}×{D['CH_H']:.1f}"],
 ['Pi bay inside', f"{D['CH_W']:.0f}×{D['CH_D'] - P['FIN_D']:.0f}×{P['BAY_H']:.0f}"],
 ['Dado depth', f"{P['DADO']:g} mm"],
 ['Shadow gap', f"{P['CAP_GAP']:g} mm"],
]
html = open('viewer_template.html').read()
html = html.replace('__GLB__', base64.b64encode(open('out/oi_speaker.glb','rb').read()).decode())
html = html.replace('__META__', json.dumps(d['groups'])).replace('__SPECS__', json.dumps(specs))
open('out/oi_speaker_viewer.html','w').write(html)
print(len(html)//1024, 'KB')
