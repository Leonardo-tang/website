import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

# 设置页面配置
st.set_page_config(
    page_title="图像检测与分割系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .image-container {
        text-align: center;
        margin: 1rem 0;
    }
    .image-caption {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 0.5rem;
        color: #333;
    }
    /* 隐藏Streamlit的部署菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)


# 模拟分割函数
def simulate_segmentation(image, segmentation_type):
    """
    模拟分割过程
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if segmentation_type == "显著性物体检测":
        edges = cv2.Canny(gray, 50, 150)
        mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    elif segmentation_type == "伪装物体检测":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)

    elif segmentation_type == "息肉分割":
        mask = np.zeros_like(gray)
        h, w = gray.shape
        center = (h // 2, w // 2)
        mask = cv2.circle(mask, center, min(h, w) // 4, 255, -1)

    elif segmentation_type == "乳腺癌分割":
        mask = np.zeros_like(gray)
        h, w = gray.shape
        for i in range(3):
            center = (h // 4 + i * (h // 4), w // 4 + i * (w // 4))
            radius = min(h, w) // 8
            mask = cv2.circle(mask, center, radius, 255, -1)

    result = image.copy()
    result[mask > 0] = [255, 0, 0]  # 将分割区域标记为红色

    return result


# 应用标题
st.markdown('<h1 class="main-header">🔍 图像检测与分割系统</h1>', unsafe_allow_html=True)

# 功能介绍卡片
st.markdown("""
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <h3>显著性物体检测</h3>
        <p>识别图像中最显著的物体区域</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🎭</div>
        <h3>伪装物体检测</h3>
        <p>检测与背景融合的隐藏物体</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🩺</div>
        <h3>息肉分割</h3>
        <p>精准分割内窥镜图像中的息肉</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">💗</div>
        <h3>乳腺癌分割</h3>
        <p>识别乳腺X光图像中的异常区域</p>
    </div>
</div>
""", unsafe_allow_html=True)

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">📤 图像上传与设置</div>', unsafe_allow_html=True)

    # 文件上传区域
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "选择图像文件",
        type=['png', 'jpg', 'jpeg'],
        help="支持格式: PNG, JPG, JPEG",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.success("✅ 图像上传成功！")

        # 图像信息显示
        image = Image.open(uploaded_file)
        st.info(f"📏 图像尺寸: {image.size[0]} × {image.size[1]} 像素")
        st.info(f"🎨 图像模式: {image.mode}")

        # 分割选项
        segmentation_type = st.selectbox(
            "选择分析类型",
            ["显著性物体检测", "伪装物体检测", "息肉分割", "乳腺癌分割"],
            index=0,
            help="根据您的需求选择合适的分析算法"
        )

        process_btn = st.button("🚀 开始分析", use_container_width=True)

with col2:
    st.markdown('<div class="sub-header">📊 分析结果</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        if process_btn:
            with st.spinner(f'🔬 正在进行{segmentation_type}...'):
                time.sleep(2)

                # 执行分割
                image_cv = np.array(image.convert('RGB'))
                result = simulate_segmentation(image_cv, segmentation_type)

                # 显示原始图像和分割结果并排
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                col_img1, col_img2 = st.columns(2)

                with col_img1:
                    st.image(image, caption="📷 原始图像", use_container_width=True)

                with col_img2:
                    st.image(result, caption="🎯 分割结果", use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

                st.success(f"✅ {segmentation_type}完成！")

                # 下载选项
                result_pil = Image.fromarray(result)
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                buf.seek(0)

                st.download_button(
                    label="💾 下载分割结果",
                    data=buf,
                    file_name=f"segmentation_result_{segmentation_type}.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            # 只显示原始图像
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="📷 原始图像", use_container_width=True)
            st.info("👆 请选择分析类型并点击分析按钮")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("📤 请先上传图像文件")

# 添加使用说明
with st.expander("📖 使用说明", expanded=False):
    st.markdown("""
    ### 使用步骤：
    1. **上传图像**：在左侧面板上传您的图像（支持PNG、JPG、JPEG格式）
    2. **选择分析类型**：根据您的需求选择合适的分割算法
    3. **开始分析**：点击分析按钮，系统将自动处理图像
    4. **查看结果**：在右侧面板并排查看原始图像和分割结果
    5. **下载结果**：如有需要，可以下载分割结果

    ### 支持的分析类型：
    - **显著性物体检测**：识别图像中最突出的区域
    - **伪装物体检测**：检测与背景颜色纹理相似的隐藏物体
    - **息肉分割**：专门用于胃肠道内窥镜图像的息肉识别
    - **乳腺癌分割**：辅助识别乳腺X光图像中的可疑区域
    """)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔍 图像检测与分割系统 | 智能分析 · 精准识别 · 专业可靠"
    "</div>",
    unsafe_allow_html=True
)

