import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: #0b0f17;
    color: #f8fafc;
}

section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263044;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1600px;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 3px;
}

.dashboard-subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}

.kpi-card {
    background: linear-gradient(145deg, #151c2b, #101622);
    border: 1px solid #263044;
    border-radius: 15px;
    padding: 20px;
    min-height: 125px;
}

.kpi-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 500;
}

.kpi-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 7px;
}

.kpi-desc {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.insight-card {
    background: #111827;
    border: 1px solid #263044;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}

.insight-title {
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 6px;
}

.insight-text {
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.5;
}

.status-online {
    color: #22c55e;
    font-weight: 600;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
}

.sidebar-subtitle {
    color: #94a3b8;
    font-size: 12px;
}

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #263044;
    border-radius: 12px;
    padding: 12px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    '<div class="sidebar-title">⚡ Attrition AI</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-subtitle">Employee Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Executive Dashboard",
        "📊 Attrition Analytics",
        "👥 Workforce Intelligence",
        "🧠 ML Model Center",
        "🔎 Data Explorer",
        "ℹ️ Project Overview"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔍 Global Filters")


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Employee Dataset",
    type=["csv"]
)


# =========================================================
# EMPTY STATE
# =========================================================

if uploaded_file is None:

    st.markdown(
        '<div class="dashboard-title">📊 Employee Attrition Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Enterprise-style workforce analytics powered by Apache Spark MLlib'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Upload your employee_attrition.csv file from the sidebar to activate the dashboard."
    )

    st.markdown(
        '<div class="section-title">🚀 Platform Capabilities</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            '<div class="insight-card">'
            '<div class="insight-title">📈 Executive Analytics</div>'
            '<div class="insight-text">'
            'Real-time workforce KPIs and attrition overview.'
            '</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            '<div class="insight-card">'
            '<div class="insight-title">👥 Workforce Intelligence</div>'
            '<div class="insight-text">'
            'Analyze demographics, roles, income and satisfaction.'
            '</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            '<div class="insight-card">'
            '<div class="insight-title">🧠 ML Model Center</div>'
            '<div class="insight-text">'
            'Monitor Spark MLlib classification performance.'
            '</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '<div class="insight-card">'
            '<div class="insight-title">🔎 Data Explorer</div>'
            '<div class="insight-text">'
            'Search, filter and download employee records.'
            '</div></div>',
            unsafe_allow_html=True
        )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(uploaded_file)

filtered_df = df.copy()


# =========================================================
# GLOBAL FILTERS
# =========================================================

if "Department" in df.columns:

    departments = st.sidebar.multiselect(
        "Department",
        sorted(df["Department"].dropna().unique()),
        default=sorted(df["Department"].dropna().unique())
    )

    filtered_df = filtered_df[
        filtered_df["Department"].isin(departments)
    ]


if "JobRole" in df.columns:

    roles = st.sidebar.multiselect(
        "Job Role",
        sorted(df["JobRole"].dropna().unique()),
        default=sorted(df["JobRole"].dropna().unique())
    )

    filtered_df = filtered_df[
        filtered_df["JobRole"].isin(roles)
    ]


if "Gender" in df.columns:

    genders = st.sidebar.multiselect(
        "Gender",
        sorted(df["Gender"].dropna().unique()),
        default=sorted(df["Gender"].dropna().unique())
    )

    filtered_df = filtered_df[
        filtered_df["Gender"].isin(genders)
    ]


if "OverTime" in df.columns:

    overtime = st.sidebar.multiselect(
        "Overtime",
        sorted(df["OverTime"].dropna().unique()),
        default=sorted(df["OverTime"].dropna().unique())
    )

    filtered_df = filtered_df[
        filtered_df["OverTime"].isin(overtime)
    ]


# =========================================================
# CALCULATIONS
# =========================================================

total_employees = len(filtered_df)

if "Attrition" in filtered_df.columns:

    attrition_count = (
        filtered_df["Attrition"]
        .astype(str)
        .str.lower()
        .eq("yes")
        .sum()
    )

else:
    attrition_count = 0


retained_count = total_employees - attrition_count

attrition_rate = (
    attrition_count / total_employees * 100
    if total_employees else 0
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="dashboard-title">📊 Employee Attrition Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Workforce analytics • Apache Spark MLlib • AWS S3 • Interactive Intelligence'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================

if page == "🏠 Executive Dashboard":

    st.markdown(
        '<div class="section-title">📈 Executive Overview</div>',
        unsafe_allow_html=True
    )

    # KPI ROW

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">TOTAL EMPLOYEES</div>'
            f'<div class="kpi-value">{total_employees:,}</div>'
            f'<div class="kpi-desc">Filtered workforce</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">ATTRITION</div>'
            f'<div class="kpi-value">{attrition_count:,}</div>'
            f'<div class="kpi-desc">Employees left</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">ATTRITION RATE</div>'
            f'<div class="kpi-value">{attrition_rate:.2f}%</div>'
            f'<div class="kpi-desc">Workforce risk indicator</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            '<div class="kpi-card">'
            '<div class="kpi-label">MODEL ACCURACY</div>'
            '<div class="kpi-value">75.59%</div>'
            '<div class="kpi-desc">Spark MLlib</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            '<div class="kpi-card">'
            '<div class="kpi-label">MODEL RECALL</div>'
            '<div class="kpi-value">77.78%</div>'
            '<div class="kpi-desc">Attrition detection</div>'
            '</div>',
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ATTRITION DONUT + DEPARTMENT

    col1, col2 = st.columns([1, 1.7])

    with col1:

        attrition_data = pd.DataFrame({
            "Status": ["Retained", "Attrition"],
            "Employees": [retained_count, attrition_count]
        })

        fig = px.pie(
            attrition_data,
            names="Status",
            values="Employees",
            hole=0.65,
            title="Workforce Retention",
            template="plotly_dark"
        )

        fig.update_layout(height=400)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        if "Department" in filtered_df.columns:

            dept = (
                filtered_df
                .groupby(["Department", "Attrition"])
                .size()
                .reset_index(name="Employees")
            )

            fig = px.bar(
                dept,
                x="Department",
                y="Employees",
                color="Attrition",
                barmode="group",
                title="Department Attrition Overview",
                template="plotly_dark"
            )

            fig.update_layout(height=400)

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # JOB ROLE + OVERTIME

    col1, col2 = st.columns(2)

    with col1:

        if "JobRole" in filtered_df.columns:

            role = (
                filtered_df
                .groupby(["JobRole", "Attrition"])
                .size()
                .reset_index(name="Employees")
            )

            fig = px.bar(
                role,
                x="JobRole",
                y="Employees",
                color="Attrition",
                barmode="group",
                title="Attrition by Job Role",
                template="plotly_dark"
            )

            fig.update_layout(
                xaxis_tickangle=-35,
                height=450
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if "OverTime" in filtered_df.columns:

            ot = (
                filtered_df
                .groupby(["OverTime", "Attrition"])
                .size()
                .reset_index(name="Employees")
            )

            fig = px.bar(
                ot,
                x="OverTime",
                y="Employees",
                color="Attrition",
                barmode="group",
                title="Overtime & Attrition",
                template="plotly_dark"
            )

            fig.update_layout(height=450)

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # BUSINESS INSIGHTS

    st.markdown(
        '<div class="section-title">💡 Automated Business Insights</div>',
        unsafe_allow_html=True
    )

    insights = []

    if attrition_rate > 15:
        insights.append(
            "⚠️ Attrition rate is relatively high and requires management attention."
        )
    else:
        insights.append(
            "✅ Current attrition rate is below the 15% monitoring threshold."
        )

    if "OverTime" in filtered_df.columns:

        overtime_attrition = filtered_df[
            filtered_df["OverTime"] == "Yes"
        ]["Attrition"].eq("Yes").sum()

        if overtime_attrition > 0:
            insights.append(
                f"⏱️ {overtime_attrition} overtime employees are associated with attrition."
            )

    if "JobRole" in filtered_df.columns:

        role_rates = (
            filtered_df
            .groupby("JobRole")["Attrition"]
            .apply(lambda x: (x == "Yes").mean() * 100)
        )

        if len(role_rates) > 0:

            highest_role = role_rates.idxmax()

            insights.append(
                f"🎯 Highest observed attrition role: {highest_role}."
            )

    for insight in insights:

        st.markdown(
            f'<div class="insight-card">'
            f'<div class="insight-text">{insight}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# =========================================================
# ATTRITION ANALYTICS
# =========================================================

elif page == "📊 Attrition Analytics":

    st.markdown(
        '<div class="section-title">📊 Attrition Analytics Center</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📌 Demographics",
        "💼 Career",
        "💰 Compensation",
        "❤️ Satisfaction"
    ])


    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            if "Age" in filtered_df.columns:

                fig = px.histogram(
                    filtered_df,
                    x="Age",
                    color="Attrition",
                    nbins=25,
                    marginal="box",
                    title="Age vs Attrition",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


        with col2:

            if "Gender" in filtered_df.columns:

                gender = (
                    filtered_df
                    .groupby(["Gender", "Attrition"])
                    .size()
                    .reset_index(name="Employees")
                )

                fig = px.bar(
                    gender,
                    x="Gender",
                    y="Employees",
                    color="Attrition",
                    barmode="group",
                    title="Gender & Attrition",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


    with tab2:

        col1, col2 = st.columns(2)

        with col1:

            if "YearsAtCompany" in filtered_df.columns:

                fig = px.box(
                    filtered_df,
                    x="Attrition",
                    y="YearsAtCompany",
                    color="Attrition",
                    title="Years at Company",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


        with col2:

            if "TotalWorkingYears" in filtered_df.columns:

                fig = px.box(
                    filtered_df,
                    x="Attrition",
                    y="TotalWorkingYears",
                    color="Attrition",
                    title="Total Working Experience",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


    with tab3:

        if "MonthlyIncome" in filtered_df.columns:

            fig = px.scatter(
                filtered_df,
                x="Age",
                y="MonthlyIncome",
                color="Attrition",
                size="YearsAtCompany",
                hover_data=["JobRole"] if "JobRole" in filtered_df.columns else None,
                title="Income vs Age vs Attrition",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with tab4:

        col1, col2 = st.columns(2)

        with col1:

            if "JobSatisfaction" in filtered_df.columns:

                sat = (
                    filtered_df
                    .groupby(["JobSatisfaction", "Attrition"])
                    .size()
                    .reset_index(name="Employees")
                )

                fig = px.bar(
                    sat,
                    x="JobSatisfaction",
                    y="Employees",
                    color="Attrition",
                    barmode="group",
                    title="Job Satisfaction",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


        with col2:

            if "EnvironmentSatisfaction" in filtered_df.columns:

                env = (
                    filtered_df
                    .groupby(["EnvironmentSatisfaction", "Attrition"])
                    .size()
                    .reset_index(name="Employees")
                )

                fig = px.bar(
                    env,
                    x="EnvironmentSatisfaction",
                    y="Employees",
                    color="Attrition",
                    barmode="group",
                    title="Environment Satisfaction",
                    template="plotly_dark"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )


# =========================================================
# WORKFORCE INTELLIGENCE
# =========================================================

elif page == "👥 Workforce Intelligence":

    st.markdown(
        '<div class="section-title">👥 Workforce Intelligence</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        if "EducationField" in filtered_df.columns:

            education = (
                filtered_df
                .groupby(["EducationField", "Attrition"])
                .size()
                .reset_index(name="Employees")
            )

            fig = px.bar(
                education,
                x="EducationField",
                y="Employees",
                color="Attrition",
                barmode="group",
                title="Education Field",
                template="plotly_dark"
            )

            fig.update_layout(xaxis_tickangle=-30)

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if "BusinessTravel" in filtered_df.columns:

            travel = (
                filtered_df
                .groupby(["BusinessTravel", "Attrition"])
                .size()
                .reset_index(name="Employees")
            )

            fig = px.bar(
                travel,
                x="BusinessTravel",
                y="Employees",
                color="Attrition",
                barmode="group",
                title="Business Travel & Attrition",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    if "MonthlyIncome" in filtered_df.columns:

        st.markdown(
            '<div class="section-title">💰 Compensation Intelligence</div>',
            unsafe_allow_html=True
        )

        fig = px.scatter(
            filtered_df,
            x="MonthlyIncome",
            y="YearsAtCompany",
            color="Attrition",
            size="Age",
            hover_data=[
                c for c in ["JobRole", "Department", "Gender"]
                if c in filtered_df.columns
            ],
            title="Income vs Tenure",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# ML MODEL CENTER
# =========================================================

elif page == "🧠 ML Model Center":

    st.markdown(
        '<div class="section-title">🧠 Spark MLlib Model Center</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Weighted Logistic Regression • Threshold = 0.4'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", "75.59%")
    c2.metric("Precision", "40.23%")
    c3.metric("Recall", "77.78%")
    c4.metric("F1", "53.03%")
    c5.metric("Threshold", "0.40")


    st.markdown("---")


    col1, col2 = st.columns(2)


    with col1:

        metrics = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ],
            "Score": [
                75.59,
                40.23,
                77.78,
                53.03
            ]
        })

        fig = px.bar(
            metrics,
            x="Metric",
            y="Score",
            text="Score",
            title="Model Performance",
            template="plotly_dark"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_range=[0, 100],
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        matrix = [
            [157, 52],
            [10, 35]
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues"
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            template="plotly_dark",
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    st.markdown(
        '<div class="section-title">🔬 Model Architecture</div>',
        unsafe_allow_html=True
    )


    architecture = pd.DataFrame({
        "Stage": [
            "Data Source",
            "Preprocessing",
            "Feature Engineering",
            "Class Balancing",
            "ML Algorithm",
            "Decision Threshold",
            "Output"
        ],
        "Technology": [
            "IBM HR Dataset",
            "Spark StringIndexer + OneHotEncoder",
            "VectorAssembler",
            "Class Weighting",
            "Weighted Logistic Regression",
            "0.40",
            "Attrition Prediction"
        ]
    })

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "The weighted model prioritizes detecting employees likely to leave. "
        "The 0.4 threshold improves attrition recall compared with a standard "
        "0.5 decision threshold."
    )


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "🔎 Data Explorer":

    st.markdown(
        '<div class="section-title">🔎 Interactive Data Explorer</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Displaying **{len(filtered_df):,}** filtered records."
    )


    search = st.text_input(
        "🔍 Search employee dataset",
        placeholder="Search any value..."
    )


    display_df = filtered_df.copy()


    if search:

        mask = display_df.astype(str).apply(
            lambda column: column.str.contains(
                search,
                case=False,
                na=False
            )
        ).any(axis=1)

        display_df = display_df[mask]


    st.dataframe(
        display_df,
        use_container_width=True,
        height=550
    )


    st.download_button(
        "⬇️ Download Filtered CSV",
        display_df.to_csv(index=False),
        "employee_attrition_filtered.csv",
        "text/csv"
    )


# =========================================================
# PROJECT OVERVIEW
# =========================================================

elif page == "ℹ️ Project Overview":

    st.markdown(
        '<div class="section-title">ℹ️ Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### Employee Attrition Prediction using Big Data Analytics

    **Data Source**

    IBM HR Employee Attrition Dataset

    **Cloud Storage**

    Amazon S3

    **Processing Engine**

    Apache Spark / PySpark

    **Machine Learning**

    Spark MLlib Weighted Logistic Regression

    **Dashboard**

    Streamlit + Plotly

    **Prediction Threshold**

    0.4
    """)


    st.markdown(
        '<div class="section-title">☁️ System Architecture</div>',
        unsafe_allow_html=True
    )


    architecture_flow = pd.DataFrame({
        "Layer": [
            "Dataset",
            "Cloud Storage",
            "Big Data Processing",
            "Machine Learning",
            "Prediction",
            "Dashboard"
        ],
        "Component": [
            "IBM HR Attrition Dataset",
            "Amazon S3",
            "Apache Spark",
            "Spark MLlib",
            "Weighted Logistic Regression",
            "Streamlit"
        ]
    })


    st.dataframe(
        architecture_flow,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        '<div class="section-title">🎯 Final Model Results</div>',
        unsafe_allow_html=True
    )


    results = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Attrition Precision",
            "Attrition Recall",
            "Attrition F1"
        ],
        "Result": [
            "75.59%",
            "40.23%",
            "77.78%",
            "53.03%"
        ]
    })


    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:12px;">'
    'Employee Attrition Intelligence Platform • Apache Spark MLlib • AWS S3 • Streamlit'
    '</div>',
    unsafe_allow_html=True
)