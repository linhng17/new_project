# app_danh_gia_du_an.py

import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError
import json 
from io import BytesIO

# Cần cài thêm thư viện python-docx nếu muốn đọc file Word
try:
    import docx
except ImportError:
    # Nếu chưa cài, sẽ cảnh báo khi người dùng tải file .docx
    docx = None 

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh giá Phương án Kinh doanh",
    layout="wide"
)

st.title("Ứng dụng Đánh giá Phương án Kinh doanh 📊")
st.markdown("Sử dụng Gemini AI để trích xuất dữ liệu, cho phép điều chỉnh thủ công và phân tích hiệu quả dự án.")

# ****************************** KHU VỰC HÀM XỬ LÝ ******************************

# 1. Hàm Trích xuất Dữ liệu bằng AI (Yêu cầu 1)
def extract_financial_data(project_text, api_key):
    """Sử dụng Gemini AI để trích xuất các thông số tài chính quan trọng."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        # Tăng cường hướng dẫn để đảm bảo đầu ra CHỈ là JSON hợp lệ
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính. Hãy trích xuất các thông tin sau từ văn bản báo cáo dự án kinh doanh được cung cấp dưới đây. 
        Đầu ra phải là một **JSON object** hoàn chỉnh và hợp lệ.

        **QUY TẮC BẮT BUỘC:**
        1. KHÔNG BAO GỒM BẤT KỲ LỜI MỞ ĐẦU, KẾT LUẬN, HOẶC VĂN BẢN NÀO KHÁC NGOÀI JSON object.
        2. Nếu không tìm thấy thông tin, hãy điền giá trị 0.0 (float) hoặc 0 (integer).
        
        Các thông tin cần trích xuất (Tên trường: Kiểu dữ liệu mong muốn):
        - Vốn đầu tư ban đầu (Capital_Investment): float (Ví dụ: 1000000.0)
        - Vòng đời dự án (Project_Life_Years): integer (Ví dụ: 5)
        - Doanh thu hàng năm (Annual_Revenue): float (Ví dụ: 500000.0)
        - Chi phí hoạt động hàng năm (Annual_Operating_Cost): float (Ví dụ: 200000.0)
        - Tỷ suất chiết khấu WACC (WACC_Rate_Percent): float (Ví dụ: 10.5)
        - Thuế suất Thu nhập Doanh nghiệp (Tax_Rate_Percent): float (Ví dụ: 20.0)
        
        Văn bản báo cáo dự án:
        ---
        {project_text}
        ---
        
        Trả lời CHỈ BẰNG JSON, KHÔNG CÓ CÁC KÝ TỰ BAO QUANH NHƯ '```json' HAY '```'.
        """
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        raw_text = response.text.strip()
        
        # Xử lý các ký tự thừa phổ biến như '```json'
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1)
        if raw_text.endswith("```"):
            raw_text = raw_text.rstrip("`")
        
        raw_text = raw_text.strip()
        
        json_data = json.loads(raw_text)
        return json_data

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error(f"Lỗi: AI không trả về dữ liệu đúng định dạng JSON. Vui lòng thử lại hoặc chỉnh sửa văn bản dự án. Chuỗi nhận được: \n\n{raw_text}")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định: {e}")
        return None

# 2 & 3. Hàm Xây dựng Dòng tiền và Tính chỉ số (Yêu cầu 2 & 3)
@st.cache_data
def calculate_cash_flow_and_metrics(data):
    """
    Xây dựng bảng dòng tiền và tính toán các chỉ số NPV, IRR, PP, DPP.
    """
    
    # Ép kiểu dữ liệu (Lấy từ dữ liệu đã qua điều chỉnh thủ công)
    I0 = data.get('Capital_Investment', 0.0)
    T = int(data.get('Project_Life_Years', 0))
    R = data.get('Annual_Revenue', 0.0)
    C = data.get('Annual_Operating_Cost', 0.0)
    WACC = data.get('WACC_Rate_Percent', 0.0) / 100.0
    Tax_Rate = data.get('Tax_Rate_Percent', 0.0) / 100.0

    if T <= 0 or I0 <= 0 or WACC <= 0:
        return None, None, "Dữ liệu đầu vào không hợp lệ (Vòng đời, Vốn đầu tư, WACC phải lớn hơn 0)."

    # Tính Dòng tiền Thuần Hoạt động hàng năm (NCF)
    EBIT = R - C
    Tax_Amount = EBIT * Tax_Rate if EBIT > 0 else 0 
    NCF_Annual = EBIT - Tax_Amount 
    
    # Xây dựng bảng dòng tiền
    years = range(T + 1)
    cash_flows = [-I0] + [NCF_Annual] * T 
    discount_factors = [1 / ((1 + WACC) ** t) for t in years]
    discounted_cash_flows = [cf * df for cf, df in zip(cash_flows, discount_factors)]

    df_cf = pd.DataFrame({
        'Năm': years,
        'Dòng tiền (CF)': cash_flows,
        'Hệ số Chiết khấu': discount_factors,
        'Dòng tiền Chiết khấu (DCF)': discounted_cash_flows
    })

    # Tính toán các chỉ số
    NPV = sum(discounted_cash_flows)
    
    try:
        IRR = np.irr(cash_flows)
    except ValueError:
        IRR = np.nan
    
    # PP (Thời gian hoàn vốn) & DPP (Thời gian hoàn vốn có chiết khấu)
    cumulative_cf = np.cumsum(cash_flows)
    pp_year = np.argmax(cumulative_cf >= 0) 
    
    if pp_year > 0 and cash_flows[pp_year] != 0:
        prev_year_cf = cumulative_cf[pp_year - 1]
        this_year_cf = cash_flows[pp_year]
        PP = pp_year - 1 + abs(prev_year_cf) / this_year_cf
    else:
        PP = np.nan
        
    cumulative_dcf = np.cumsum(discounted_cash_flows)
    dpp_year = np.argmax(cumulative_dcf >= 0)
    
    if dpp_year > 0 and discounted_cash_flows[dpp_year] != 0:
        prev_year_dcf = cumulative_dcf[dpp_year - 1]
        this_year_dcf = discounted_cash_flows[dpp_year]
        DPP = dpp_year - 1 + abs(prev_year_dcf) / this_year_dcf
    else:
        DPP = np.nan
        
    metrics = {
        "Vốn Đầu tư Ban đầu": I0,
        "Vòng đời Dự án (Năm)": T,
        "WACC": WACC,
        "Lãi suất IRR": IRR,
        "Giá trị Hiện tại Ròng (NPV)": NPV,
        "Thời gian Hoàn vốn (PP)": PP,
        "Thời gian Hoàn vốn Chiết khấu (DPP)": DPP,
        "Dòng tiền Thuần Hoạt động hàng năm (NCF)": NCF_Annual,
    }
    
    return df_cf, metrics, None


# 4. Hàm Phân tích Chỉ số bằng AI (Yêu cầu 4)
# (Không thay đổi)
def analyze_metrics_with_ai(metrics_data, api_key):
    """Gửi các chỉ số đánh giá dự án đến AI để phân tích."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        metrics_text = "\n".join([f"- {k}: {v:,.2f}" if isinstance(v, (int, float)) and v is not np.nan else f"- {k}: {v}" for k, v in metrics_data.items()])

        prompt = f"""
        Bạn là một chuyên gia phân tích dự án đầu tư. Dựa trên các chỉ số hiệu quả dự án sau, hãy đưa ra một đánh giá chuyên sâu và khách quan (khoảng 3-4 đoạn). 
        
        **Đánh giá cần tập trung vào:**
        1. **Khả năng sinh lời:** Nhận xét về NPV và so sánh IRR với WACC (chỉ số WACC đã được cung cấp).
        2. **Tính khả thi:** Kết luận dự án có nên được chấp nhận hay không (Dựa trên NPV > 0 và IRR > WACC).
        3. **Rủi ro thanh khoản:** Đánh giá thời gian hoàn vốn (PP và DPP) so với vòng đời dự án.

        Các chỉ số tài chính của dự án:
        ---
        {metrics_text}
        ---
        
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định khi yêu cầu phân tích AI: {e}"

# ****************************** KHU VỰC GIAO DIỆN STREAMLIT ******************************

# --- Thanh bên cấu hình API ---
with st.sidebar:
    st.header("🔑 Cấu hình API")
    api_key_input = st.text_input(
        "Nhập Khóa API Google Gemini",
        type="password",
        help="Khóa API này cần được lưu trữ bảo mật (Ví dụ: trong Streamlit Secrets)."
    )
    if not api_key_input:
        st.warning("Vui lòng nhập Khóa API Gemini để sử dụng chức năng AI.")

# --- Chức năng 1: Tải File/Nhập Văn bản ---
st.subheader("1. Tải lên và Trích xuất Dữ liệu Dự án (AI)")
uploaded_file = st.file_uploader(
    "Tải lên file Word (.docx) hoặc Text (.txt) có chứa thông tin dự án.",
    type=['docx', 'txt']
)

project_text_area = st.text_area(
    "Hoặc dán nội dung báo cáo dự án vào đây:",
    height=300,
    placeholder="Dán toàn bộ văn bản dự án của khách hàng vào đây. Đảm bảo có các thông tin về Vốn đầu tư, Vòng đời, Doanh thu, Chi phí, WACC, Thuế."
)

project_text = ""
if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        project_text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        if docx:
            try:
                doc = docx.Document(BytesIO(uploaded_file.read()))
                project_text = "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                st.error(f"Lỗi đọc file Word: {e}")
                project_text = ""
        else:
            st.warning("Cảnh báo: Cần cài đặt thư viện 'python-docx' để đọc file Word.")
            project_text = ""
    
    if project_text:
        st.text_area("Nội dung được trích xuất từ file:", value=project_text, height=100)
    
if project_text_area:
    project_text = project_text_area

# --- Nút Lọc Dữ liệu ---
if st.button("🚀 Lọc Dữ liệu Dự án bằng AI", disabled=not (project_text and api_key_input)):
    
    if not api_key_input:
        st.error("Vui lòng cung cấp Khóa API Gemini.")
    elif not project_text:
        st.error("Vui lòng tải file hoặc dán nội dung dự án.")
    else:
        # Xóa session state cũ
        if 'extracted_data' in st.session_state:
            del st.session_state['extracted_data']
            
        with st.spinner('Đang gửi văn bản và chờ AI trích xuất thông tin...'):
            extracted_data = extract_financial_data(project_text, api_key_input)
            
            if extracted_data:
                # Lưu dữ liệu thô vào session state
                st.session_state['extracted_data'] = extracted_data
                st.success("Trích xuất dữ liệu thành công! ✅")
            else:
                st.session_state['extracted_data'] = None

# --- Chức năng 1.5: Điều chỉnh Thủ công ---
if 'extracted_data' in st.session_state and st.session_state['extracted_data']:
    
    st.markdown("---")
    st.subheader("1.5. Điều chỉnh Thủ công Dữ liệu Dự án")
    st.warning("Vui lòng kiểm tra và điều chỉnh các giá trị trích xuất (hoặc điền thủ công nếu AI thất bại) trước khi tính toán.")
    
    data = st.session_state['extracted_data']
    
    # Thiết lập giá trị mặc định nếu AI trích xuất thất bại hoặc trả về 0
    I0 = data.get('Capital_Investment', 0.0)
    T = data.get('Project_Life_Years', 0)
    R = data.get('Annual_Revenue', 0.0)
    C = data.get('Annual_Operating_Cost', 0.0)
    WACC_rate = data.get('WACC_Rate_Percent', 0.0)
    Tax_rate = data.get('Tax_Rate_Percent', 0.0)
    
    # Sử dụng st.columns và st.number_input để người dùng điều chỉnh
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data['Capital_Investment'] = st.number_input(
            "💰 Vốn Đầu tư Ban đầu (I0)", value=float(I0), min_value=0.0, step=1000000.0, format="%.0f", key="input_I0"
        )
        data['Annual_Revenue'] = st.number_input(
            "📈 Doanh thu Hàng năm", value=float(R), min_value=0.0, step=100000.0, format="%.0f", key="input_R"
        )
    
    with col2:
        data['Project_Life_Years'] = st.number_input(
            "⏳ Vòng đời Dự án (Năm)", value=int(T), min_value=1, step=1, key="input_T"
        )
        data['Annual_Operating_Cost'] = st.number_input(
            "📉 Chi phí Hàng năm", value=float(C), min_value=0.0, step=100000.0, format="%.0f", key="input_C"
        )
    
    with col3:
        data['WACC_Rate_Percent'] = st.number_input(
            "⚖️ Tỷ suất Chiết khấu (WACC, %)", value=float(WACC_rate), min_value=0.0, max_value=100.0, step=0.1, key="input_WACC"
        )
        data['Tax_Rate_Percent'] = st.number_input(
            "🏛️ Thuế suất Thu nhập DN (%)", value=float(Tax_rate), min_value=0.0, max_value=100.0, step=0.1, key="input_Tax"
        )

    # Cập nhật session state với dữ liệu đã điều chỉnh
    st.session_state['extracted_data'] = data
    
    # --- Chức năng Tính toán (Chạy sau khi có dữ liệu đã điều chỉnh) ---
    df_cash_flow, metrics, error = calculate_cash_flow_and_metrics(st.session_state['extracted_data'])

    if error:
        st.error(f"Lỗi tính toán: {error}")
    elif df_cash_flow is not None:
        
        st.markdown("---")
        
        ## 2. Bảng Dòng tiền của Dự án
        st.subheader("2. Bảng Dòng tiền và Dòng tiền Chiết khấu")
        st.dataframe(df_cash_flow.style.format({
            'Dòng tiền (CF)': '{:,.0f}',
            'Hệ số Chiết khấu': '{:.4f}',
            'Dòng tiền Chiết khấu (DCF)': '{:,.0f}'
        }), use_container_width=True, hide_index=True)

        st.markdown("---")
        
        ## 3. Các Chỉ số Đánh giá Hiệu quả Dự án
        st.subheader("3. Các Chỉ số Đánh giá Hiệu quả Dự án")

        col_npv, col_irr, col_pp, col_dpp = st.columns(4)
        
        def display_metric(col, label, value, format_str=""):
            if value is not np.nan and value is not None:
                if label == "NPV":
                    delta = "Dự án Khả thi (Chấp nhận)" if value > 0 else "Dự án Không khả thi (Từ chối)"
                    delta_color = "inverse" if value > 0 else "off"
                    col.metric(f"💰 {label}", f"{value:{format_str}}", delta=delta, delta_color=delta_color)
                elif label == "IRR":
                    col.metric(f"📈 {label} (Internal Rate of Return)", f"{value:{format_str}}")
                elif "Hoàn vốn" in label:
                     col.metric(f"⏳ {label}", f"{value:.2f} năm")
                else:
                    col.metric(label, f"{value:{format_str}}")
            else:
                 col.metric(label, "N/A")

        display_metric(col_npv, "NPV", metrics["Giá trị Hiện tại Ròng (NPV)"], ",.0f")
        display_metric(col_irr, "IRR", metrics["Lãi suất IRR"], ".2%")
        display_metric(col_pp, "Thời gian Hoàn vốn (PP)", metrics["Thời gian Hoàn vốn (PP)"])
        display_metric(col_dpp, "Thời gian Hoàn vốn Chiết khấu (DPP)", metrics["Thời gian Hoàn vốn Chiết khấu (DPP)"])

        st.markdown(f"**Tỷ suất Chiết khấu (WACC):** {metrics['WACC']:.2%} | **Vòng đời Dự án:** {metrics['Vòng đời Dự án (Năm)']} năm")

        st.markdown("---")
        
        ## 4. Phân tích Chỉ số bằng AI
        st.subheader("4. Yêu cầu AI Phân tích Chỉ số")

        if st.button("🤖 Yêu cầu AI Phân tích Hiệu quả Dự án"):
             if api_key_input:
                with st.spinner('Đang gửi các chỉ số và chờ Gemini phân tích...'):
                    ai_analysis_result = analyze_metrics_with_ai(metrics, api_key_input)
                    st.markdown("### Kết quả Phân tích từ Gemini AI")
                    st.info(ai_analysis_result)
             else:
                st.error("Vui lòng nhập Khóa API Gemini để thực hiện phân tích.")
                
else:
    st.info("Vui lòng tải lên file hoặc dán nội dung dự án và bấm nút **Lọc Dữ liệu Dự án bằng AI** để bắt đầu.")
