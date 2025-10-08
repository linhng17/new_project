# app_danh_gia_du_an.py

import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError
import json # Dùng để xử lý dữ liệu JSON trích xuất từ AI
from io import BytesIO

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh giá Phương án Kinh doanh",
    layout="wide"
)

st.title("Ứng dụng Đánh giá Phương án Kinh doanh 📊")
st.markdown("Sử dụng Gemini AI để trích xuất dữ liệu và phân tích hiệu quả dự án.")

# ****************************** KHU VỰC HÀM XỬ LÝ ******************************

# 1. Hàm Trích xuất Dữ liệu bằng AI (Yêu cầu 1)
def extract_financial_data(project_text, api_key):
    """Sử dụng Gemini AI để trích xuất các thông số tài chính quan trọng."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        # Prompt chi tiết yêu cầu AI trả về dữ liệu dưới định dạng JSON
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính. Hãy trích xuất các thông tin sau từ văn bản báo cáo dự án kinh doanh được cung cấp dưới đây. 
        Đầu ra phải là một **JSON object** hoàn chỉnh.
        Nếu không tìm thấy thông tin, hãy điền giá trị 0 hoặc chuỗi rỗng.
        
        Các thông tin cần trích xuất (Tên trường: Kiểu dữ liệu mong muốn):
        - Vốn đầu tư ban đầu (Capital_Investment): float (Ví dụ: 1000000)
        - Vòng đời dự án (Project_Life_Years): integer (Ví dụ: 5)
        - Doanh thu hàng năm (Annual_Revenue): float (Ví dụ: 500000)
        - Chi phí hoạt động hàng năm (Annual_Operating_Cost): float (Ví dụ: 200000)
        - Tỷ suất chiết khấu WACC (WACC_Rate_Percent): float (Ví dụ: 10.5 -> tương ứng 0.105)
        - Thuế suất Thu nhập Doanh nghiệp (Tax_Rate_Percent): float (Ví dụ: 20 -> tương ứng 0.20)
        
        Văn bản báo cáo dự án:
        ---
        {project_text}
        ---
        
        Trả lời CHỈ BẰNG JSON, không có bất kỳ lời mở đầu hay kết luận nào.
        """
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        # Xử lý chuỗi JSON và chuyển thành Dict/Object
        json_data = json.loads(response.text.strip())
        return json_data

    except APIError as e:
        st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Lỗi: AI không trả về dữ liệu đúng định dạng JSON. Vui lòng thử lại hoặc chỉnh sửa văn bản dự án.")
        return None
    except Exception as e:
        st.error(f"Đã xảy ra lỗi không xác định: {e}")
        return None

# 2 & 3. Hàm Xây dựng Dòng tiền và Tính chỉ số (Yêu cầu 2 & 3)
@st.cache_data
def calculate_cash_flow_and_metrics(data):
    """
    Xây dựng bảng dòng tiền và tính toán các chỉ số NPV, IRR, PP, DPP.
    Giả định: Dòng tiền hoạt động hàng năm là Cân bằng (Revenue - Cost) * (1 - Tax)
    """
    
    # Ép kiểu dữ liệu
    I0 = data.get('Capital_Investment', 0.0)
    T = int(data.get('Project_Life_Years', 0))
    R = data.get('Annual_Revenue', 0.0)
    C = data.get('Annual_Operating_Cost', 0.0)
    WACC = data.get('WACC_Rate_Percent', 0.0) / 100.0
    Tax_Rate = data.get('Tax_Rate_Percent', 0.0) / 100.0

    if T <= 0 or I0 <= 0 or WACC <= 0:
        return None, {"NPV": np.nan, "IRR": np.nan, "PP": np.nan, "DPP": np.nan}, "Dữ liệu đầu vào không hợp lệ (Vòng đời, Vốn đầu tư, WACC phải lớn hơn 0)."

    # Tính Dòng tiền Thuần Hoạt động hàng năm (NCF)
    # Giả định đơn giản: NCF = (Doanh thu - Chi phí) * (1 - Thuế)
    EBIT = R - C
    Tax_Amount = EBIT * Tax_Rate if EBIT > 0 else 0 
    NCF_Annual = EBIT - Tax_Amount # Lợi nhuận sau thuế (Đã bao gồm khấu hao nếu Chi phí là Chi phí tiền mặt)
    
    # Xây dựng bảng dòng tiền
    years = range(T + 1)
    cash_flows = [-I0] + [NCF_Annual] * T # Năm 0 là vốn đầu tư, các năm còn lại là NCF
    discount_factors = [1 / ((1 + WACC) ** t) for t in years]
    discounted_cash_flows = [cf * df for cf, df in zip(cash_flows, discount_factors)]

    df_cf = pd.DataFrame({
        'Năm': years,
        'Dòng tiền (CF)': cash_flows,
        'Hệ số Chiết khấu': discount_factors,
        'Dòng tiền Chiết khấu (DCF)': discounted_cash_flows
    })

    # Tính toán các chỉ số
    # NPV
    NPV = sum(discounted_cash_flows)
    
    # IRR (Sử dụng numpy.irr)
    try:
        IRR = np.irr(cash_flows)
    except ValueError:
        IRR = np.nan
    
    # PP (Thời gian hoàn vốn) & DPP (Thời gian hoàn vốn có chiết khấu)
    
    # Tính dòng tiền lũy kế (Cho PP)
    cumulative_cf = np.cumsum(cash_flows)
    pp_year = np.argmax(cumulative_cf >= 0) 
    
    if pp_year > 0:
        prev_year_cf = cumulative_cf[pp_year - 1]
        this_year_cf = cash_flows[pp_year]
        PP = pp_year - 1 + abs(prev_year_cf) / this_year_cf
    else:
        PP = np.nan
        
    # Tính dòng tiền chiết khấu lũy kế (Cho DPP)
    cumulative_dcf = np.cumsum(discounted_cash_flows)
    dpp_year = np.argmax(cumulative_dcf >= 0)
    
    if dpp_year > 0:
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
def analyze_metrics_with_ai(metrics_data, api_key):
    """Gửi các chỉ số đánh giá dự án đến AI để phân tích."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Chuyển metrics_data sang định dạng dễ đọc cho AI
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
    # Nếu là file .txt, đọc nội dung
    if uploaded_file.type == "text/plain":
        project_text = uploaded_file.read().decode("utf-8")
    # Nếu là file .docx (Cần thư viện python-docx), giả định đã cài
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            import docx
            doc = docx.Document(BytesIO(uploaded_file.read()))
            project_text = "\n".join([p.text for p in doc.paragraphs])
        except ImportError:
            st.warning("Cảnh báo: Cần cài đặt thư viện 'python-docx' (`pip install python-docx`) để đọc file Word.")
            project_text = "Lỗi đọc file Word: Vui lòng dán nội dung vào ô bên dưới."
    st.text_area("Nội dung được trích xuất từ file:", value=project_text, height=100)
    
# Ưu tiên nội dung từ text_area nếu nó được điền
if project_text_area:
    project_text = project_text_area

# --- Nút Lọc Dữ liệu ---
if st.button("🚀 Lọc Dữ liệu Dự án bằng AI", disabled=not (project_text and api_key_input)):
    
    if not api_key_input:
        st.error("Vui lòng cung cấp Khóa API Gemini.")
    elif not project_text:
        st.error("Vui lòng tải file hoặc dán nội dung dự án.")
    else:
        with st.spinner('Đang gửi văn bản và chờ AI trích xuất thông tin...'):
            extracted_data = extract_financial_data(project_text, api_key_input)
            
            if extracted_data:
                # Lưu dữ liệu vào session state để sử dụng sau
                st.session_state['extracted_data'] = extracted_data
                st.success("Trích xuất dữ liệu thành công! ✅")

                # Hiển thị dữ liệu đã trích xuất
                st.subheader("Thông tin Dự án đã Trích xuất")
                df_data = pd.DataFrame(extracted_data.items(), columns=["Chỉ tiêu", "Giá trị"])
                df_data['Giá trị'] = df_data['Giá trị'].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and x >= 1000 else f"{x:.2f}" if isinstance(x, float) else x)
                st.table(df_data)

# --- Các bước tiếp theo ---
if 'extracted_data' in st.session_state and st.session_state['extracted_data']:
    
    extracted_data = st.session_state['extracted_data']
    
    # Thực hiện bước 2 & 3
    df_cash_flow, metrics, error = calculate_cash_flow_and_metrics(extracted_data)

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
        
        # Hàm hiển thị metric
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
