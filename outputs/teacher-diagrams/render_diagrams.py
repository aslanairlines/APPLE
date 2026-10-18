from pathlib import Path
import subprocess

OUT = Path(__file__).parent

BASE = '''
graph [rankdir=TB, bgcolor="white", pad="0.35", nodesep="0.30", ranksep="0.48", fontname="DejaVu Sans", fontsize=22, fontcolor="#17243B", labelloc=t, labeljust=l, dpi=130];
node [shape=box, style="rounded,filled", fillcolor="#F2F5F9", color="#CBD5E1", fontname="DejaVu Sans", fontsize=14, fontcolor="#17243B", margin="0.18,0.13", penwidth=1.1];
edge [color="#64748B", arrowsize=0.75, penwidth=1.4, fontname="DejaVu Sans", fontsize=11, fontcolor="#475569"];
'''

GRAPHS = {
"01-teacher-architecture": r'''
label="1  /  TEACHER ARCHITECTURE\nSource identity + target appearance guide the generator";
source [label="SOURCE IMAGE\nWhose identity?", fillcolor="#DBEAFE", color="#60A5FA"];
target [label="TARGET IMAGE\nWhich pose and appearance?", fillcolor="#FEF3C7", color="#FBBF24"];
{rank=same; source; target;}
arc [label="InsightFace / ArcFace branch\n512-dimensional recognition vector", fillcolor="#EFF6FF"];
prep [label="Align + parse face\nGrayscale face; whiten selected regions", fillcolor="#EFF6FF"];
eva [label="EVA-CLIP vision encoder\nGlobal vector: 768 dimensions\nFive intermediate token sequences", fillcolor="#EFF6FF"];
id [label="IDFormer\nFuse recognition + global EVA vector\nAttend to intermediate EVA features\nOutput: 32 identity tokens × 2048", fillcolor="#DBEAFE", color="#60A5FA"];
cond [label="TARGET PREPROCESSING\nBlur facial region\nAdd face + iris landmarks", fillcolor="#FFFBEB"];
vae [label="Frozen VAE encoder\nTarget condition tokens", fillcolor="#FEF3C7", color="#FBBF24"];
source -> arc;
source -> prep -> eva;
arc -> id [label="512 features"];
eva -> id [label="768 features + visual tokens"];
target -> cond -> vae;
subgraph cluster_flux {
  label="FLUX + TEACHER LoRA"; fontsize=17; color="#A78BFA"; style="rounded"; bgcolor="#FAF8FF"; margin=18;
  dual [label="19 dual-stream transformer blocks\nText and image streams interact", fillcolor="#EDE9FE", color="#A78BFA"];
  single [label="38 single-stream transformer blocks\nCombined text + image tokens", fillcolor="#EDE9FE", color="#A78BFA"];
  velocity [label="Output projection\nPredicted latent velocity", fillcolor="#EDE9FE", color="#A78BFA"];
  dual -> single -> velocity;
}
other [label="OTHER INPUTS\nCurrent noisy latent + timestep\nText encoded with CLIP + T5", fillcolor="#F1F5F9"];
vae -> dual [label="Image condition"];
id -> dual [label="Identity cross-attention\nevery 2 blocks", color="#3B82F6"];
id -> single [label="Identity cross-attention\nevery 4 blocks", color="#3B82F6"];
other -> dual;
''',
"02-teacher-training": r'''
label="2  /  TEACHER TRAINING\nSource and target show the SAME person";
source [label="SOURCE: ALICE\nPhoto 1", fillcolor="#DBEAFE", color="#60A5FA"];
target [label="TARGET: ALICE\nPhoto 2 — the correct answer", fillcolor="#FEF3C7", color="#FBBF24"];
{rank=same; source; target;}
id [label="ArcFace + EVA-CLIP → IDFormer\nSource identity tokens\nUsually precomputed / cached", fillcolor="#DBEAFE", color="#60A5FA"];
cond [label="Blur + face / iris landmarks\nFrozen VAE encoder\nTarget condition tokens", fillcolor="#FEF3C7", color="#FBBF24"];
clean [label="Frozen VAE encoder\nClean target latent: z", fillcolor="#FEF3C7", color="#FBBF24"];
noise [label="Sample noise ε and timestep t"];
mix [label="Mix target latent with noise\nz_t = (1 − t) z + t ε"];
flux [label="FLUX + TEACHER LoRA\nOne forward pass at sampled timestep\nAlso receives text embeddings + timestep", fillcolor="#EDE9FE", color="#A78BFA"];
pred [label="Predicted velocity: v", fillcolor="#EDE9FE", color="#A78BFA"];
fm [label="FLOW-MATCHING LOSS\nCompare v with ε − z", fillcolor="#FCE7F3", color="#F472B6"];
est [label="Estimated clean image\nẑ = z_t − t v → VAE decoder"];
aux [label="OPTIONAL AUXILIARY LOSSES\nIdentity: output vs source\nLPIPS: output vs original target\nApplied at selected timesteps", fillcolor="#FCE7F3", color="#F472B6"];
loss [label="Total loss → backpropagation\nUpdate selected trainable weights", fillcolor="#DCFCE7", color="#4ADE80"];
source -> id -> flux;
target -> cond -> flux;
target -> clean -> mix;
noise -> mix -> flux;
flux -> pred;
pred -> fm;
pred -> est -> aux;
fm -> loss;
aux -> loss;
''',
"03-teacher-testing": r'''
label="3  /  TESTING: FACE SWAPPING\nSource and target can show DIFFERENT people";
source [label="SOURCE: ALICE\nIdentity to transfer", fillcolor="#DBEAFE", color="#60A5FA"];
target [label="TARGET: BOB\nPose, expression, lighting, appearance", fillcolor="#FEF3C7", color="#FBBF24"];
{rank=same; source; target;}
id [label="ArcFace + EVA-CLIP → IDFormer\nAlice identity tokens", fillcolor="#DBEAFE", color="#60A5FA"];
cond [label="Blur + face / iris landmarks\nVAE encoder → target condition", fillcolor="#FEF3C7", color="#FBBF24"];
start [label="STARTING LATENT\nRandom Gaussian noise\nOR optional target inversion"];
flux [label="TRAINED FLUX + TEACHER LoRA\nInputs: current latent, source identity,\ntarget condition, text and timestep\nWeights stay fixed", fillcolor="#EDE9FE", color="#A78BFA"];
step [label="Predict velocity\nScheduler updates the latent", fillcolor="#EDE9FE", color="#A78BFA"];
check [shape=diamond, style=filled, label="More steps?", fillcolor="#F1F5F9", margin="0.12,0.08"];
decode [label="Frozen VAE decoder\nFinal latent → RGB image"];
out [label="SWAPPED IMAGE\nAlice identity + Bob attributes\nIntended result", fillcolor="#DCFCE7", color="#4ADE80"];
source -> id -> flux;
target -> cond -> flux;
start -> flux;
flux -> step -> check;
check -> flux [label="YES: repeat with updated latent\n28 generation steps by default", color="#7C3AED"];
check -> decode [label="NO: finished"];
decode -> out;
note [shape=note, label="Optional inversion\nEncode original Bob image with VAE\nReverse flow using target condition\nUse resulting latent as the start", fillcolor="#FFFBEB", color="#FBBF24"];
target -> note [style=dashed];
note -> start [style=dashed];
'''
}

for name, body in GRAPHS.items():
    dot = OUT / f"{name}.dot"
    dot.write_text("digraph G {\n" + BASE + body + "\n}\n")
    for fmt in ("svg", "png"):
        subprocess.run(["dot", f"-T{fmt}", str(dot), "-o", str(OUT / f"{name}.{fmt}")], check=True)
    print(OUT / f"{name}.png")
