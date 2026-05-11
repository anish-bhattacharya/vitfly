#!/usr/bin/env python3
"""Generate draw.io XML for 6 branch architecture diagrams."""

import os, subprocess

OUT = os.path.join(os.path.dirname(__file__), 'figures', 'arch')
os.makedirs(OUT, exist_ok=True)

DRAWIO = '/usr/bin/drawio'

TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="drawio" version="30.0.0">
  <diagram name="Page-1" id="page1">
    <mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="800" pageHeight="350" background="#ffffff">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- Title -->
        <mxCell id="t1" value="Branch {BRANCH}: {TITLE}" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=11;fontStyle=1;fontFamily=SimSun;fontColor=#000;" vertex="1" parent="1">
          <mxGeometry x="100" y="5" width="400" height="25" as="geometry" />
        </mxCell>

        <!-- Input -->
        <mxCell id="in" value="Depth&#xa;60×90" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F0F0F0;strokeColor=#000;strokeWidth=1.5;fontSize=10;fontFamily=SimSun;fontColor=#000;" vertex="1" parent="1">
          <mxGeometry x="20" y="100" width="65" height="50" as="geometry" />
        </mxCell>

        {LAYERS}

        <!-- Concat -->
        <mxCell id="cat" value="Concat&#xa;+vel+quat" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E0E0E0;strokeColor=#000;strokeWidth=1.2;fontSize=8;fontFamily=SimSun;fontColor=#000;" vertex="1" parent="1">
          <mxGeometry x="{CAT_X}" y="105" width="55" height="40" as="geometry" />
        </mxCell>

        <!-- Temporal Head -->
        <mxCell id="tem" value="{TEMP_LABEL}" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#D0D0D0;strokeColor=#000;strokeWidth=1.5;fontSize=9;fontFamily=SimSun;fontColor=#000;" vertex="1" parent="1">
          <mxGeometry x="{TEMP_X}" y="90" width="80" height="70" as="geometry" />
        </mxCell>

        <!-- Output -->
        <mxCell id="out" value="Velocity&#xa;(vx,vy,vz)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F0F0F0;strokeColor=#000;strokeWidth=1.5;fontSize=9;fontFamily=SimSun;fontColor=#000;" vertex="1" parent="1">
          <mxGeometry x="{OUT_X}" y="100" width="65" height="50" as="geometry" />
        </mxCell>

        <!-- Arrows -->
        {ARROWS}
        <mxCell id="a_in_enc" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#000;strokeWidth=1.2;" edge="1" parent="1" source="in" target="{ENC_ID}">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a_enc_cat" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#000;strokeWidth=1.2;" edge="1" parent="1" source="{ENC_ID}" target="cat">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a_cat_tem" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#000;strokeWidth=1.2;" edge="1" parent="1" source="cat" target="tem">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a_tem_out" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#000;strokeWidth=1.2;" edge="1" parent="1" source="tem" target="out">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

BRANCHES = {
 'A': {
   'title': 'VMamba+LSTM (0.97M params)',
   'layers': [
     ('Conv 3×3 32ch', '#E0E0E0', '0.8'),
     ('Conv 3×3 64ch', '#D4D4D4', '0.8'),
     ('Conv 3×3 128ch', '#C8C8C8', '0.8'),
     ('SS2D 4-dir\nscan 256ch', '#B0B0B0', '1.2'),
   ],
   'enc_note': '4608-dim',
   'temp': 'LSTM×3\nh=128',
 },
 'B': {
   'title': 'MambaVision+SSM (2.61M params)',
   'layers': [
     ('Stem 7×7 s4', '#E0E0E0', '0.8'),
     ('DWConv+MLP\n×2', '#D4D4D4', '0.8'),
     ('DWConv+MLP\n×2', '#C8C8C8', '0.8'),
     ('DWConv+MLP\n×2', '#B8B8B8', '0.8'),
   ],
   'enc_note': '512-dim',
   'temp': 'SSM\nd=16×2',
 },
 'B+': {
   'title': 'MambaVision+Mamba-3 (2.55M params)',
   'layers': [
     ('Stem 7×7 s4', '#E0E0E0', '0.8'),
     ('DWConv+MLP\n×2', '#D4D4D4', '0.8'),
     ('DWConv+MLP\n×2', '#C8C8C8', '0.8'),
     ('DWConv+MLP\n×2', '#B8B8B8', '0.8'),
   ],
   'enc_note': '512-dim',
   'temp': 'Mamba-3\nExp-trap',
 },
 'C': {
   'title': 'CNN+Mamba-3 (2.41M params)',
   'layers': [
     ('Conv 3×3\n32ch s2', '#E0E0E0', '0.8'),
     ('Conv 3×3\n64ch s2', '#D4D4D4', '0.8'),
     ('Conv 3×3\n128ch s2', '#C8C8C8', '0.8'),
     ('Conv 3×3 256ch\n+GAP', '#B0B0B0', '1.2'),
   ],
   'enc_note': '1.81M encoder',
   'temp': 'Mamba-3\nd=32',
 },
 'D': {
   'title': 'STH-Mamba (2.60M params)',
   'layers': [
     ('Conv 3×3\n32ch', '#E0E0E0', '0.8'),
     ('Conv 3×3\n64ch', '#D4D4D4', '0.8'),
     ('Conv 3×3\n128ch', '#C8C8C8', '0.8'),
     ('ST-Mamba\nscan 256ch', '#B0B0B0', '1.2'),
   ],
   'enc_note': '1.80M encoder',
   'temp': 'Mamba-2\nSSD d=128',
 },
 'E': {
   'title': 'DecisionMamba (2.19M params)',
   'layers': [
     ('Conv 3×3\n32ch s2', '#E0E0E0', '0.8'),
     ('Conv 3×3\n64ch s2', '#D4D4D4', '0.8'),
     ('Conv 3×3\n128ch s2', '#C8C8C8', '0.8'),
     ('Conv 3×3 256ch\n+AP', '#B0B0B0', '1.2'),
   ],
   'enc_note': '455K encoder',
   'temp': 'SSM d=16\n×2',
 },
}

def build_xml(branch, spec):
    layers_xml = ''
    arrows_xml = ''
    enc_id = ''
    prev_id = 'in'
    x = 100
    layer_ids = []

    for i, (name, fc, sw) in enumerate(spec['layers']):
        lid = f'enc_{i}'
        layer_ids.append(lid)
        ly = 85 + (3-i) * 18  # stagger for 3D stack effect
        lx = 100 + i * 5       # slight x offset for depth
        layers_xml += f'''        <mxCell id="{lid}" value="{name}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fc};strokeColor=#000;strokeWidth={sw};fontSize=7;fontFamily=SimSun;fontColor=#000;" vertex="1" parent="1">
          <mxGeometry x="{lx}" y="{ly}" width="80" height="50" as="geometry" />
        </mxCell>
'''
        if i > 0:
            arrows_xml += f'''        <mxCell id="a_l{i}" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#000;strokeWidth=0.8;" edge="1" parent="1" source="{layer_ids[i-1]}" target="{lid}">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
'''

    enc_id = layer_ids[-1] if layer_ids else 'in'

    cat_x = 210
    temp_x = 290
    out_x = 400

    # Encoder note
    layers_xml += f'''        <mxCell id="enc_note" value="{spec['enc_note']}" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=7;fontFamily=Times New Roman;fontColor=#666;fontStyle=italic;" vertex="1" parent="1">
          <mxGeometry x="100" y="140" width="80" height="15" as="geometry" />
        </mxCell>
'''

    return TEMPLATE.format(
        BRANCH=branch, TITLE=spec['title'],
        LAYERS=layers_xml, ARROWS=arrows_xml,
        CAT_X=cat_x, TEMP_X=temp_x, OUT_X=out_x,
        ENC_ID=enc_id,
        TEMP_LABEL=spec['temp'],
    )

def main():
    for branch, spec in BRANCHES.items():
        xml = build_xml(branch, spec)
        path = os.path.join(OUT, f'arch_branch_{branch}.drawio')
        with open(path, 'w') as f:
            f.write(xml)
        print(f'{path} written')

        # Export to PDF
        pdf_path = os.path.join(OUT, f'arch_branch_{branch}.pdf')
        subprocess.run([
            DRAWIO, '--no-sandbox', '-x', '-f', 'pdf',
            '-o', pdf_path, path
        ], capture_output=True)
        sz = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
        print(f'  PDF: {pdf_path} ({sz:,} bytes)')

if __name__ == '__main__':
    main()
