import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="한국/미국 주식 수익률 비교",
    page_icon="📈",
    layout="wide"
)

st.title("📈 한국/미국 주요 주식 수익률 비교 웹앱")
st.caption("yfinance 데이터를 활용해 한국과 미국 주요 주식의 가격, 수익률, 변동성 등을 비교합니다.")


# =========================
# 주요 종목 목록
# =========================
KOREA_STOCKS = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "LG에너지솔루션": "373220.KS",
    "셀트리온": "068270.KS",
    "삼성바이오로직스": "207940.KS",
    "POSCO홀딩스": "005490.KS",
    "KB금융": "105560.KS",
    "신한지주": "055550.KS",
    "LG화학": "051910.KS",
    "삼성SDI": "006400.KS",
    "현대모비스": "012330.KS",
}

US_STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Alphabet Class A": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Tesla": "TSLA",
    "Netflix": "NFLX",
    "Berkshire Hathaway B": "BRK-B",
    "JPMorgan Chase": "JPM",
    "Visa": "V",
    "Johnson & Johnson": "JNJ",
    "Procter & Gamble": "PG",
    "Coca-Cola": "KO",
    "Walmart": "WMT",
}

INDEX_TICKERS = {
    "KOSPI": "^KS11",
    "KOSDAQ": "^KQ11",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
}


# =========================
# 보조 함수
# =========================
def get_name_from_ticker(ticker, name_map):
    for name, code in name_map.items():
        if code == ticker:
            return name
    return ticker


@st.cache_data(ttl=3600)
def load_price_data(tickers, start_date, end_date):
    """
    yfinance에서 종가 데이터를 불러오는 함수
    auto_adjust=True를 사용하면 배당, 액면분할 등을 반영한 가격을 받을 수 있음
    """
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if data.empty:
        return pd.DataFrame()

    # 여러 종목일 때와 한 종목일 때 구조가 다를 수 있으므로 처리
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        else:
            close = data.xs("Close", axis=1, level=0)
    else:
        close = data[["Close"]]
        close.columns = tickers

    close = close.dropna(how="all")
    return close


def calculate_metrics(price_df):
    """
    가격 데이터로부터 총수익률, 연환산 수익률, 변동성, 최대낙폭, 샤프비율 계산
    """
    if price_df.empty:
        return pd.DataFrame()

    daily_returns = price_df.pct_change().dropna(how="all")

    metrics = []

    for ticker in price_df.columns:
        series = price_df[ticker].dropna()

        if len(series) < 2:
            continue

        start_price = series.iloc[0]
        end_price = series.iloc[-1]

        total_return = end_price / start_price - 1

        days = (series.index[-1] - series.index[0]).days
        years = days / 365.25 if days > 0 else np.nan

        if years and years > 0:
            annual_return = (end_price / start_price) ** (1 / years) - 1
        else:
            annual_return = np.nan

        if ticker in daily_returns.columns:
            volatility = daily_returns[ticker].std() * np.sqrt(252)
            sharpe = annual_return / volatility if volatility and volatility != 0 else np.nan
        else:
            volatility = np.nan
            sharpe = np.nan

        cumulative_max = series.cummax()
        drawdown = series / cumulative_max - 1
        max_drawdown = drawdown.min()

        metrics.append({
            "Ticker": ticker,
            "시작가": start_price,
            "종가": end_price,
            "총수익률": total_return,
            "연환산수익률": annual_return,
            "연환산변동성": volatility,
            "최대낙폭": max_drawdown,
            "샤프비율": sharpe
        })

    return pd.DataFrame(metrics)


def make_cumulative_return(price_df):
    """
    누적수익률 계산
    """
    if price_df.empty:
        return pd.DataFrame()

    normalized = price_df / price_df.iloc[0]
    cumulative_return = normalized - 1
    return cumulative_return


def format_percent_df(df, percent_columns):
    formatted = df.copy()
    for col in percent_columns:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
    return formatted


# =========================
# 사이드바 입력
# =========================
st.sidebar.header("⚙️ 분석 설정")

market_option = st.sidebar.multiselect(
    "분석할 시장 선택",
    ["한국 주식", "미국 주식", "주요 지수"],
    default=["한국 주식", "미국 주식"]
)

today = date.today()
default_start = today - timedelta(days=365)

start_date = st.sidebar.date_input("시작일", value=default_start)
end_date = st.sidebar.date_input("종료일", value=today)

if start_date >= end_date:
    st.error("시작일은 종료일보다 빨라야 합니다.")
    st.stop()


selected_tickers = []
ticker_name_map = {}

if "한국 주식" in market_option:
    korea_selected_names = st.sidebar.multiselect(
        "한국 주요 주식 선택",
        list(KOREA_STOCKS.keys()),
        default=["삼성전자", "SK하이닉스", "NAVER"]
    )
    for name in korea_selected_names:
        ticker = KOREA_STOCKS[name]
        selected_tickers.append(ticker)
        ticker_name_map[ticker] = name

if "미국 주식" in market_option:
    us_selected_names = st.sidebar.multiselect(
        "미국 주요 주식 선택",
        list(US_STOCKS.keys()),
        default=["Apple", "Microsoft", "NVIDIA"]
    )
    for name in us_selected_names:
        ticker = US_STOCKS[name]
        selected_tickers.append(ticker)
        ticker_name_map[ticker] = name

if "주요 지수" in market_option:
    index_selected_names = st.sidebar.multiselect(
        "주요 지수 선택",
        list(INDEX_TICKERS.keys()),
        default=["KOSPI", "S&P 500", "NASDAQ"]
    )
    for name in index_selected_names:
        ticker = INDEX_TICKERS[name]
        selected_tickers.append(ticker)
        ticker_name_map[ticker] = name


st.sidebar.divider()

custom_tickers_input = st.sidebar.text_input(
    "직접 티커 추가",
    placeholder="예: 005930.KS, AAPL, TSLA"
)

if custom_tickers_input:
    custom_tickers = [
        ticker.strip().upper()
        for ticker in custom_tickers_input.split(",")
        if ticker.strip()
    ]
    for ticker in custom_tickers:
        selected_tickers.append(ticker)
        ticker_name_map[ticker] = ticker

selected_tickers = list(dict.fromkeys(selected_tickers))

st.sidebar.info(
    """
    한국 주식은 보통 티커 뒤에 `.KS` 또는 `.KQ`를 붙입니다.

    예시:
    - 삼성전자: `005930.KS`
    - 카카오: `035720.KS`
    - 에코프로비엠: `247540.KQ`
    """
)


# =========================
# 데이터 로드
# =========================
if not selected_tickers:
    st.warning("왼쪽 사이드바에서 분석할 종목을 선택하세요.")
    st.stop()

with st.spinner("yfinance에서 데이터를 불러오는 중입니다..."):
    price_df = load_price_data(selected_tickers, start_date, end_date)

if price_df.empty:
    st.error("데이터를 불러오지 못했습니다. 티커나 날짜 범위를 확인하세요.")
    st.stop()

# 컬럼 순서 정리
available_columns = [col for col in selected_tickers if col in price_df.columns]
price_df = price_df[available_columns]

# 이름 표시용 컬럼명
display_columns = {
    ticker: f"{ticker_name_map.get(ticker, ticker)} ({ticker})"
    for ticker in price_df.columns
}

price_display_df = price_df.rename(columns=display_columns)

cumulative_return_df = make_cumulative_return(price_df)
cumulative_return_display_df = cumulative_return_df.rename(columns=display_columns)

daily_return_df = price_df.pct_change().dropna(how="all")
daily_return_display_df = daily_return_df.rename(columns=display_columns)

metrics_df = calculate_metrics(price_df)

if not metrics_df.empty:
    metrics_df["종목명"] = metrics_df["Ticker"].map(lambda x: ticker_name_map.get(x, x))
    metrics_df = metrics_df[
        [
            "종목명",
            "Ticker",
            "시작가",
            "종가",
            "총수익률",
            "연환산수익률",
            "연환산변동성",
            "최대낙폭",
            "샤프비율"
        ]
    ]


# =========================
# 핵심 요약
# =========================
st.subheader("📌 선택 종목")
st.write(", ".join([display_columns[t] for t in price_df.columns]))

col1, col2, col3, col4 = st.columns(4)

if not metrics_df.empty:
    best_return_row = metrics_df.loc[metrics_df["총수익률"].idxmax()]
    worst_return_row = metrics_df.loc[metrics_df["총수익률"].idxmin()]
    lowest_vol_row = metrics_df.loc[metrics_df["연환산변동성"].idxmin()]
    best_sharpe_row = metrics_df.loc[metrics_df["샤프비율"].idxmax()]

    with col1:
        st.metric(
            "최고 총수익률",
            f"{best_return_row['종목명']}",
            f"{best_return_row['총수익률']:.2%}"
        )

    with col2:
        st.metric(
            "최저 총수익률",
            f"{worst_return_row['종목명']}",
            f"{worst_return_row['총수익률']:.2%}"
        )

    with col3:
        st.metric(
            "가장 낮은 변동성",
            f"{lowest_vol_row['종목명']}",
            f"{lowest_vol_row['연환산변동성']:.2%}"
        )

    with col4:
        st.metric(
            "최고 샤프비율",
            f"{best_sharpe_row['종목명']}",
            f"{best_sharpe_row['샤프비율']:.2f}"
        )


# =========================
# 탭 구성
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "누적수익률 비교",
        "가격 차트",
        "일별 수익률",
        "성과 지표",
        "원본 데이터"
    ]
)


# =========================
# 탭 1: 누적수익률 비교
# =========================
with tab1:
    st.subheader("📈 누적수익률 비교")

    fig = px.line(
        cumulative_return_display_df,
        x=cumulative_return_display_df.index,
        y=cumulative_return_display_df.columns,
        title="선택 종목 누적수익률 비교",
        labels={
            "value": "누적수익률",
            "index": "날짜",
            "variable": "종목"
        }
    )

    fig.update_layout(
        hovermode="x unified",
        yaxis_tickformat=".0%",
        legend_title_text="종목"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        """
        누적수익률은 선택한 시작일의 가격을 기준으로 얼마나 상승 또는 하락했는지를 보여줍니다.

        예를 들어 누적수익률이 20%라면, 시작일에 투자했을 때 현재 약 20% 수익이 났다는 뜻입니다.
        """
    )


# =========================
# 탭 2: 가격 차트
# =========================
with tab2:
    st.subheader("💰 가격 차트")

    chart_type = st.radio(
        "가격 차트 표시 방식",
        ["일반 가격", "시작일=100으로 정규화"],
        horizontal=True
    )

    if chart_type == "일반 가격":
        plot_df = price_display_df
        y_label = "가격"
        title = "선택 종목 가격 추이"
    else:
        plot_df = price_df / price_df.iloc[0] * 100
        plot_df = plot_df.rename(columns=display_columns)
        y_label = "정규화 가격"
        title = "선택 종목 가격 추이 시작일=100"

    fig_price = px.line(
        plot_df,
        x=plot_df.index,
        y=plot_df.columns,
        title=title,
        labels={
            "value": y_label,
            "index": "날짜",
            "variable": "종목"
        }
    )

    fig_price.update_layout(
        hovermode="x unified",
        legend_title_text="종목"
    )

    st.plotly_chart(fig_price, use_container_width=True)

    st.warning(
        """
        한국 주식은 원화, 미국 주식은 달러 기준 가격입니다.
        따라서 서로 다른 통화의 가격을 직접 비교하기보다는 수익률 또는 정규화 가격으로 비교하는 것이 더 적절합니다.
        """
    )


# =========================
# 탭 3: 일별 수익률
# =========================
with tab3:
    st.subheader("📊 일별 수익률 분석")

    selected_for_hist = st.selectbox(
        "히스토그램으로 볼 종목 선택",
        list(daily_return_display_df.columns)
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(
            daily_return_display_df,
            x=selected_for_hist,
            nbins=50,
            title=f"{selected_for_hist} 일별 수익률 분포",
            labels={selected_for_hist: "일별 수익률"}
        )
        fig_hist.update_layout(xaxis_tickformat=".1%")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box = px.box(
            daily_return_display_df,
            y=daily_return_display_df.columns,
            title="종목별 일별 수익률 박스플롯",
            labels={
                "value": "일별 수익률",
                "variable": "종목"
            }
        )
        fig_box.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig_box, use_container_width=True)

    st.info(
        """
        일별 수익률 분포를 보면 종목의 변동성이 어느 정도인지 파악할 수 있습니다.

        분포가 넓게 퍼져 있을수록 하루하루 가격 변동이 큰 편입니다.
        """
    )


# =========================
# 탭 4: 성과 지표
# =========================
with tab4:
    st.subheader("🏆 성과 지표 비교")

    if metrics_df.empty:
        st.warning("성과 지표를 계산할 수 없습니다.")
    else:
        formatted_metrics_df = metrics_df.copy()

        percent_cols = ["총수익률", "연환산수익률", "연환산변동성", "최대낙폭"]
        formatted_metrics_df = format_percent_df(formatted_metrics_df, percent_cols)

        for col in ["시작가", "종가"]:
            formatted_metrics_df[col] = formatted_metrics_df[col].apply(
                lambda x: f"{x:,.2f}" if pd.notnull(x) else "-"
            )

        formatted_metrics_df["샤프비율"] = formatted_metrics_df["샤프비율"].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "-"
        )

        st.dataframe(
            formatted_metrics_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        metric_to_plot = st.selectbox(
            "막대그래프로 비교할 지표 선택",
            ["총수익률", "연환산수익률", "연환산변동성", "최대낙폭", "샤프비율"]
        )

        plot_metric_df = metrics_df.copy()
        plot_metric_df["표시명"] = plot_metric_df["종목명"] + " (" + plot_metric_df["Ticker"] + ")"

        fig_bar = px.bar(
            plot_metric_df,
            x="표시명",
            y=metric_to_plot,
            title=f"{metric_to_plot} 비교",
            labels={
                "표시명": "종목",
                metric_to_plot: metric_to_plot
            },
            text_auto=True
        )

        if metric_to_plot != "샤프비율":
            fig_bar.update_layout(yaxis_tickformat=".1%")

        fig_bar.update_layout(xaxis_tickangle=-30)

        st.plotly_chart(fig_bar, use_container_width=True)

        with st.expander("성과 지표 설명 보기"):
            st.markdown(
                """
                - **총수익률**: 분석 기간 동안 가격이 얼마나 상승 또는 하락했는지 나타냅니다.
                - **연환산수익률**: 분석 기간의 수익률을 1년 기준으로 환산한 값입니다.
                - **연환산변동성**: 일별 수익률의 표준편차를 1년 기준으로 환산한 값입니다.
                - **최대낙폭, MDD**: 고점 대비 가장 크게 하락한 비율입니다.
                - **샤프비율**: 수익률을 변동성으로 나눈 값입니다. 여기서는 무위험수익률을 0으로 가정했습니다.
                """
            )


# =========================
# 탭 5: 원본 데이터
# =========================
with tab5:
    st.subheader("🧾 원본 가격 데이터")

    st.dataframe(
        price_display_df.sort_index(ascending=False),
        use_container_width=True
    )

    csv_data = price_display_df.to_csv(index=True).encode("utf-8-sig")

    st.download_button(
        label="CSV 다운로드",
        data=csv_data,
        file_name="stock_price_data.csv",
        mime="text/csv"
    )


# =========================
# 푸터
# =========================
st.divider()
st.caption(
    """
    데이터 출처: Yahoo Finance via yfinance  
    본 웹앱은 데이터 분석 학습용 예제이며 투자 판단의 근거로 사용해서는 안 됩니다.
    """
)
