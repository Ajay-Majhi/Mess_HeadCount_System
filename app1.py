import streamlit as st
import sqlite3
import pandas as pd
from datetime import date
import io
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from keras_facenet import FaceNet

DB_PATH = "mess_headcount.db"


# ── Database helpers ─────────────────────────────────────────────────────────

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS personnel (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rank TEXT NOT NULL,
            unit TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS headcount (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            meal TEXT NOT NULL CHECK(meal IN ('Breakfast', 'Lunch', 'Dinner')),
            personnel_id INTEGER NOT NULL REFERENCES personnel(id),
            recorded_by TEXT NOT NULL DEFAULT 'Admin',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(date, meal, personnel_id)
        );

        CREATE TABLE IF NOT EXISTS video_headcount (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            date TEXT NOT NULL,
            meal TEXT NOT NULL,
            total_count INTEGER NOT NULL,
            peak_count INTEGER NOT NULL,
            avg_count REAL NOT NULL,
            frames_analysed INTEGER NOT NULL,
            recorded_by TEXT NOT NULL DEFAULT 'Admin',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def get_personnel(active_only=True):
    conn = get_connection()
    where = "WHERE active = 1" if active_only else ""
    df = pd.read_sql_query(
        f"SELECT id, name, rank, unit, active FROM personnel {where} ORDER BY name",
        conn,
    )
    conn.close()
    return df


def add_personnel(name, rank, unit):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO personnel (name, rank, unit) VALUES (?, ?, ?)",
            (name.strip(), rank.strip(), unit.strip()),
        )
        conn.commit()
        return True, "Personnel added successfully."
    except sqlite3.IntegrityError as e:
        return False, str(e)
    finally:
        conn.close()


def toggle_personnel(pid, active):
    conn = get_connection()
    conn.execute("UPDATE personnel SET active = ? WHERE id = ?", (active, pid))
    conn.commit()
    conn.close()


def record_headcount(record_date, meal, personnel_ids, recorded_by):
    conn = get_connection()
    inserted = 0
    for pid in personnel_ids:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO headcount (date, meal, personnel_id, recorded_by) VALUES (?, ?, ?, ?)",
                (str(record_date), meal, pid, recorded_by),
            )
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass
    conn.commit()
    conn.close()
    return inserted


def get_headcount_for_meal(record_date, meal):
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT p.name, p.rank, p.unit
        FROM headcount h
        JOIN personnel p ON p.id = h.personnel_id
        WHERE h.date = ? AND h.meal = ?
        ORDER BY p.name
        """,
        conn,
        params=(str(record_date), meal),
    )
    conn.close()
    return df


def save_video_result(filename, vid_date, meal, total, peak, avg, frames, recorded_by):
    conn = get_connection()
    conn.execute(
        """INSERT INTO video_headcount
           (filename, date, meal, total_count, peak_count, avg_count, frames_analysed, recorded_by)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (filename, str(vid_date), meal, total, peak, round(avg, 2), frames, recorded_by),
    )
    conn.commit()
    conn.close()


def get_video_history():
    conn = get_connection()
    df = pd.read_sql_query(
        """SELECT filename AS "File", date AS "Date", meal AS "Meal",
                  peak_count AS "Peak Count", avg_count AS "Avg Count",
                  frames_analysed AS "Frames Analysed", recorded_by AS "Recorded By",
                  created_at AS "Uploaded At"
           FROM video_headcount ORDER BY created_at DESC""",
        conn,
    )
    conn.close()
    return df


def get_summary(start_date, end_date):
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT h.date, h.meal, COUNT(h.id) AS headcount
        FROM headcount h
        WHERE h.date BETWEEN ? AND ?
        GROUP BY h.date, h.meal
        ORDER BY h.date, h.meal
        """,
        conn,
        params=(str(start_date), str(end_date)),
    )
    conn.close()
    return df


def get_detailed_report(start_date, end_date):
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT h.date AS Date, h.meal AS Meal,
               p.rank AS Rank, p.name AS Name, p.unit AS Unit,
               h.recorded_by AS "Recorded By"
        FROM headcount h
        JOIN personnel p ON p.id = h.personnel_id
        WHERE h.date BETWEEN ? AND ?
        ORDER BY h.date, h.meal, p.name
        """,
        conn,
        params=(str(start_date), str(end_date)),
    )
    conn.close()
    return df


def get_attendance_by_person(start_date, end_date):
    conn = get_connection()
    df = pd.read_sql_query(
        """
        SELECT p.rank AS Rank, p.name AS Name, p.unit AS Unit,
               SUM(CASE WHEN h.meal = 'Breakfast' THEN 1 ELSE 0 END) AS Breakfast,
               SUM(CASE WHEN h.meal = 'Lunch' THEN 1 ELSE 0 END) AS Lunch,
               SUM(CASE WHEN h.meal = 'Dinner' THEN 1 ELSE 0 END) AS Dinner,
               COUNT(h.id) AS Total
        FROM personnel p
        LEFT JOIN headcount h ON h.personnel_id = p.id AND h.date BETWEEN ? AND ?
        WHERE p.active = 1
        GROUP BY p.id
        ORDER BY Total DESC
        """,
        conn,
        params=(str(start_date), str(end_date)),
    )
    conn.close()
    return df


def to_excel(dfs: dict):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()



# ── Video processing (YOLO + DeepFace + FaceNet) ─────────────────────────────

embedder = FaceNet()
yolo_model = YOLO("yolov8n.pt")   # use YOLO11 face model

def get_embedding(face_img):
    face = cv2.resize(face_img,(160,160))
    face = np.expand_dims(face,axis=0)
    embedding = embedder.embeddings(face)
    return embedding[0]


def match_face(embedding, database, threshold=0.7):

    if len(database)==0:
        return False

    for db_emb in database:

        dist = np.linalg.norm(embedding-db_emb)

        if dist < threshold:
            return True

    return False


def process_video(video_path, sample_every=15, progress_cb=None):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    frame_no = 0

    unique_faces = []
    counts=[]
    sample_imgs=[]

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_no % sample_every == 0:

            results = yolo_model(frame)

            faces_in_frame=0

            annotated = frame.copy()

            for r in results:

                boxes=r.boxes.xyxy.cpu().numpy()

                for box in boxes:

                    x1,y1,x2,y2 = map(int,box)

                    face = frame[y1:y2,x1:x2]

                    if face.size==0:
                        continue

                    try:

                        emb = get_embedding(face)

                        known = match_face(emb,unique_faces)

                        if not known:

                            unique_faces.append(emb)

                        faces_in_frame+=1

                        cv2.rectangle(
                            annotated,
                            (x1,y1),
                            (x2,y2),
                            (0,255,0),
                            2
                        )

                    except:
                        pass


            counts.append((frame_no,faces_in_frame))

            cv2.putText(
                annotated,
                f"Faces: {faces_in_frame}  Unique: {len(unique_faces)}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            _,buf=cv2.imencode(".jpg",annotated)

            sample_imgs.append((frame_no,buf.tobytes()))

            if progress_cb and total_frames>0:

                progress_cb(
                    min(frame_no/total_frames,1.0)
                )

        frame_no+=1


    cap.release()

    if progress_cb:
        progress_cb(1.0)


    return counts,sample_imgs,fps,len(unique_faces)


# ── App layout ────────────────────────────────────────────────────────────────

init_db()

st.set_page_config(
    page_title="Mess Headcount System",
    page_icon="._.",
    layout="wide",
)

st.title(" Mess Headcount System")
st.markdown("---")

page = st.sidebar.selectbox(
    "Navigation",
    [
        " Record Headcount",
        " Video Headcount",
        " Personnel Management",
        " Reports & Summary",
    ],
)

# ── Record Headcount ──────────────────────────────────────────────────────────
if page == " Record Headcount":
    st.header("Record Headcount")

    col1, col2, col3 = st.columns(3)
    with col1:
        record_date = st.date_input("Date", value=date.today())
    with col2:
        meal = st.selectbox("Meal", ["Breakfast", "Lunch", "Dinner"])
    with col3:
        recorded_by = st.text_input("Recorded By", value="Admin")

    personnel_df = get_personnel(active_only=True)

    if personnel_df.empty:
        st.warning("No active personnel found. Please add personnel first.")
    else:
        already_df = get_headcount_for_meal(record_date, meal)
        already_ids = set()
        if not already_df.empty:
            conn = get_connection()
            id_df = pd.read_sql_query(
                "SELECT h.personnel_id FROM headcount h WHERE h.date = ? AND h.meal = ?",
                conn, params=(str(record_date), meal),
            )
            conn.close()
            already_ids = set(id_df["personnel_id"].tolist())

        st.subheader(f"Select personnel present for {meal} on {record_date}")

        all_ids = personnel_df["id"].tolist()

        if "selected_ids" not in st.session_state:
            st.session_state.selected_ids = list(already_ids)

        check_col1, check_col2 = st.columns(2)
        with check_col1:
            if st.button("Select All"):
                st.session_state.selected_ids = all_ids
        with check_col2:
            if st.button("Clear All"):
                st.session_state.selected_ids = []

        selected_ids = []
        for unit in sorted(personnel_df["unit"].unique()):
            unit_df = personnel_df[personnel_df["unit"] == unit]
            st.markdown(f"**Unit: {unit}**")
            cols = st.columns(3)
            for i, (_, row) in enumerate(unit_df.iterrows()):
                with cols[i % 3]:
                    checked = st.checkbox(
                        f"{row['rank']} {row['name']}",
                        value=row["id"] in st.session_state.selected_ids,
                        key=f"chk_{row['id']}_{record_date}_{meal}",
                    )
                    if checked:
                        selected_ids.append(row["id"])
            st.markdown("---")

        if st.button(" Save Headcount", type="primary"):
            if not recorded_by.strip():
                st.error("Please enter who is recording.")
            else:
                conn = get_connection()
                conn.execute(
                    "DELETE FROM headcount WHERE date = ? AND meal = ?",
                    (str(record_date), meal),
                )
                conn.commit()
                conn.close()
                if selected_ids:
                    record_headcount(record_date, meal, selected_ids, recorded_by.strip())
                st.success(f"Headcount saved: {len(selected_ids)} personnel for {meal} on {record_date}.")
                st.session_state.selected_ids = selected_ids

        st.subheader(f"Current {meal} Attendance ({record_date})")
        present_df = get_headcount_for_meal(record_date, meal)
        if present_df.empty:
            st.info("No headcount recorded yet for this meal.")
        else:
            st.dataframe(present_df, use_container_width=True)
            st.metric("Total Present", len(present_df))

# ── Video Headcount ───────────────────────────────────────────────────────────
elif page == "🎥 Video Headcount":
    st.header("Video Headcount Extraction")
    st.write(
        "Upload a mess / canteen video and the system will automatically count "
        "the number of people visible and report the total headcount."
    )

    with st.expander("ℹ How it works", expanded=False):
        st.markdown("""
        - The video is sampled every N frames (configurable).
        - Each sampled frame is analysed using the **HOG (Histogram of Oriented Gradients)**
          person detector built into OpenCV — no internet or GPU required.
        - Detected persons are highlighted with green bounding boxes.
        - The **peak count** (maximum persons detected in any single frame) is used as the
          estimated headcount for that meal session.
        - Results are saved to the database and can be exported.
        """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        vid_date = st.date_input("Date", value=date.today(), key="vid_date")
    with col2:
        vid_meal = st.selectbox("Meal", ["Breakfast", "Lunch", "Dinner"], key="vid_meal")
    with col3:
        sample_every = st.slider("Sample every N frames", 10, 120, 30, step=10)
    with col4:
        vid_recorded_by = st.text_input("Recorded By", value="Admin", key="vid_recorded_by")

    uploaded = st.file_uploader(
        "Upload video file",
        type=["mp4", "avi", "mov", "mkv", "wmv"],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV",
    )

    if uploaded is not None:
        st.video(uploaded)

        if st.button(" Analyse Video", type="primary"):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded.name)[1]
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                progress_bar = st.progress(0.0, text="Analysing video…")

                def update_progress(p):
                    progress_bar.progress(p, text=f"Analysing… {int(p*100)}%")

                with st.spinner("Processing frames — this may take a moment…"):
                    counts, sample_imgs, fps = process_video(
                        tmp_path, sample_every=sample_every, progress_cb=update_progress
                    )

                progress_bar.empty()

                if not counts:
                    st.error("Could not read frames from the video. Please try a different file.")
                else:
                    frame_nos = [c[0] for c in counts]
                    people_counts = [c[1] for c in counts]

                    peak = max(people_counts)
                    avg = sum(people_counts) / len(people_counts)
                    total_frames_analysed = len(counts)

                    save_video_result(
                        uploaded.name, vid_date, vid_meal,
                        peak, peak, avg, total_frames_analysed,
                        vid_recorded_by,
                    )

                    st.markdown("---")
                    st.subheader(" Analysis Results")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Peak Headcount", peak)
                    m2.metric("Average per Frame", f"{avg:.1f}")
                    m3.metric("Frames Analysed", total_frames_analysed)
                    m4.metric("Video FPS", f"{fps:.0f}")

                    chart_df = pd.DataFrame(
                        {"Frame": frame_nos, "People Detected": people_counts}
                    ).set_index("Frame")
                    st.subheader("People Count Over Time")
                    st.line_chart(chart_df)

                    st.subheader("Annotated Sample Frames")
                    st.caption("Green boxes = detected persons")

                    max_display = 9
                    display_imgs = sample_imgs[:: max(1, len(sample_imgs) // max_display)][:max_display]
                    n_cols = 3
                    rows = [display_imgs[i: i + n_cols] for i in range(0, len(display_imgs), n_cols)]
                    for row in rows:
                        img_cols = st.columns(n_cols)
                        for col, (fno, img_bytes) in zip(img_cols, row):
                            count_at = people_counts[frame_nos.index(fno)]
                            col.image(
                                img_bytes,
                                caption=f"Frame {fno} — {count_at} person(s)",
                                use_container_width=True,
                            )

                    st.success(
                        f" Analysis complete. Estimated headcount: **{peak} people** "
                        f"(peak across {total_frames_analysed} sampled frames). Result saved."
                    )

            finally:
                os.unlink(tmp_path)

    st.markdown("---")
    st.subheader(" Previous Video Analysis History")
    hist_df = get_video_history()
    if hist_df.empty:
        st.info("No video analyses recorded yet.")
    else:
        st.dataframe(hist_df, use_container_width=True)

# ── Personnel Management ──────────────────────────────────────────────────────
elif page == "👥 Personnel Management":
    st.header("Personnel Management")

    tab1, tab2 = st.tabs(["Add Personnel", "Manage Existing"])

    with tab1:
        st.subheader("Add New Personnel")
        with st.form("add_personnel_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                name = st.text_input("Full Name *")
            with col2:
                rank = st.text_input("Rank *")
            with col3:
                unit = st.text_input("Unit / Department *")
            submitted = st.form_submit_button("Add Personnel", type="primary")
            if submitted:
                if not name or not rank or not unit:
                    st.error("All fields are required.")
                else:
                    ok, msg = add_personnel(name, rank, unit)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

    with tab2:
        st.subheader("All Personnel")
        show_inactive = st.checkbox("Show inactive personnel")
        df = get_personnel(active_only=not show_inactive)

        if df.empty:
            st.info("No personnel records found.")
        else:
            df["Status"] = df["active"].apply(lambda x: "✅ Active" if x else "❌ Inactive")
            st.dataframe(
                df[["name", "rank", "unit", "Status"]].rename(
                    columns={"name": "Name", "rank": "Rank", "unit": "Unit"}
                ),
                use_container_width=True,
            )

            st.markdown("---")
            st.subheader("Activate / Deactivate Personnel")
            pid_to_name = {
                f"{row['rank']} {row['name']} ({row['unit']})": row["id"]
                for _, row in df.iterrows()
            }
            selected_label = st.selectbox("Select Personnel", list(pid_to_name.keys()))
            selected_pid = pid_to_name[selected_label]
            selected_row = df[df["id"] == selected_pid].iloc[0]

            if selected_row["active"] == 1:
                if st.button("Deactivate", type="secondary"):
                    toggle_personnel(selected_pid, 0)
                    st.success("Personnel deactivated.")
                    st.rerun()
            else:
                if st.button("Activate", type="primary"):
                    toggle_personnel(selected_pid, 1)
                    st.success("Personnel activated.")
                    st.rerun()

# ── Reports & Summary ─────────────────────────────────────────────────────────
elif page == " Reports & Summary":
    st.header("Reports & Summary")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From Date", value=date.today().replace(day=1))
    with col2:
        end_date = st.date_input("To Date", value=date.today())

    if start_date > end_date:
        st.error("Start date must be before end date.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Daily Summary", "Attendance by Person", "Detailed Log"])

    with tab1:
        st.subheader("Daily Meal Headcount Summary")
        summary_df = get_summary(start_date, end_date)
        if summary_df.empty:
            st.info("No data found for the selected date range.")
        else:
            pivot = summary_df.pivot_table(
                index="date", columns="meal", values="headcount", fill_value=0
            ).reset_index()
            for m in ["Breakfast", "Lunch", "Dinner"]:
                if m not in pivot.columns:
                    pivot[m] = 0
            pivot["Total"] = pivot[["Breakfast", "Lunch", "Dinner"]].sum(axis=1)
            pivot = pivot.rename(columns={"date": "Date"})
            st.dataframe(pivot, use_container_width=True)

            st.subheader("Headcount Trend")
            chart_data = summary_df.pivot_table(
                index="date", columns="meal", values="headcount", fill_value=0
            )
            st.line_chart(chart_data)

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Breakfast", f"{pivot['Breakfast'].mean():.1f}" if "Breakfast" in pivot else "0")
            c2.metric("Avg Lunch", f"{pivot['Lunch'].mean():.1f}" if "Lunch" in pivot else "0")
            c3.metric("Avg Dinner", f"{pivot['Dinner'].mean():.1f}" if "Dinner" in pivot else "0")

    with tab2:
        st.subheader("Attendance by Personnel")
        person_df = get_attendance_by_person(start_date, end_date)
        if person_df.empty:
            st.info("No data found.")
        else:
            st.dataframe(person_df, use_container_width=True)

    with tab3:
        st.subheader("Detailed Attendance Log")
        detail_df = get_detailed_report(start_date, end_date)
        if detail_df.empty:
            st.info("No records found for the selected date range.")
        else:
            st.dataframe(detail_df, use_container_width=True)
            st.markdown(f"**Total Records:** {len(detail_df)}")

    st.markdown("---")
    st.subheader("Export to Excel")
    if st.button(" Download Excel Report"):
        excel_data = to_excel({
            "Daily Summary": get_summary(start_date, end_date),
            "By Personnel": get_attendance_by_person(start_date, end_date),
            "Detailed Log": get_detailed_report(start_date, end_date),
            "Video Analysis": get_video_history(),
        })
        st.download_button(
            label="Click here to download",
            data=excel_data,
            file_name=f"mess_headcount_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
