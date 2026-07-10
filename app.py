
import streamlit as st
from pathlib import Path
import time
import re

# Import your backend
from HitchAI import run

ROOT = Path(__file__).parent.resolve()

def render_markdown_with_images(md: str):
    parts = re.split(r'(!\[.*?\]\(.*?\))', md)

    for part in parts:

        if part.startswith("!["):

            match = re.search(r'!\[(.*?)\]\((.*?)\)', part)

            if not match:
                continue

            alt = match.group(1)
            img_path = match.group(2)

            p = (ROOT / img_path).resolve()

            if not p.exists():
                p = (ROOT / "images" / Path(img_path).name).resolve()

            if p.exists():
                st.image(str(p), caption=alt, width="stretch")
            else:
                st.warning(f"Image not found: {img_path}")

        else:
            if part.strip():
                st.markdown(part)


st.set_page_config(
    page_title="Blog Auto Writer",
    page_icon="📝",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top:1.5rem;}
.stButton>button{
    width:100%;
    border-radius:10px;
    height:48px;
}
.card{
    padding:1rem;
    border:1px solid #444;
    border-radius:12px;
    margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)

BLOG_DIR = Path(".")
IMAGE_DIR = Path("images")

def list_blogs():
    return sorted(BLOG_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

def read_blog(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except:
        return ""

def extract_images(md):
    return re.findall(r'!\[.*?\]\((.*?)\)', md)

if "selected" not in st.session_state:
    st.session_state.selected = None

with st.sidebar:
    st.title("📚 Blogs")
    q = st.text_input("Search")

    if st.button("🔄 Refresh"):
        st.rerun()

    st.divider()

    blogs = list_blogs()

    if not blogs:
        st.info("No blogs generated yet.")
    
    for blog in blogs:
        if q.find("README")!=-1: continue
        if q.lower() not in blog.stem.lower():
            continue
        if st.button(blog.stem, key=blog.name):
            st.session_state.selected = blog

st.title("🤖 HitchAI - Agentic Blog Auto Writer")

topic = st.text_input(
    "Topic",
    placeholder="Attention Mechanism in Transformers"
)

if st.button("🚀 Generate Blog", type="primary"):

    if topic.strip() == "":
        st.warning("Enter a topic.")
    else:
        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Planning...")
            progress.progress(15)

            start = time.time()

            out = run(topic)

            progress.progress(80)
            status.info("Loading generated blog...")

            newest = list_blogs()

            if newest:
                st.session_state.selected = newest[0]

            progress.progress(100)
            status.success(f"Completed in {time.time()-start:.1f}s")

        except Exception as e:
            st.exception(e)

st.divider()

if st.session_state.selected:

    blog = st.session_state.selected

    md = read_blog(blog)

    words = len(md.split())
    read = max(1, words//200)

    c1,c2,c3=st.columns(3)
    c1.metric("Words",words)
    c2.metric("Reading Time",f"{read} min")
    c3.metric("Images",len(extract_images(md)))

    tab1,tab2,tab3=st.tabs(["📖 Blog","🖼 Images","⬇ Download"])

    with tab1:
        render_markdown_with_images(md)

    with tab2:
        imgs=extract_images(md)
        if not imgs:
            st.info("No images.")
        for img in imgs:
            p=Path(img)
            if not p.exists():
                p=IMAGE_DIR/p.name
            if p.exists():
                st.image(str(p),use_container_width=True)
            else:
                st.warning(f"Missing image: {img}")

    with tab3:
        st.download_button(
            "Download Markdown",
            md,
            file_name=blog.name,
            mime="text/markdown"
        )

else:
    st.info("Generate or select a blog from the sidebar.")
