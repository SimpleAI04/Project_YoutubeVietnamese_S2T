import os
import subprocess
import shutil
import re
import time

MIN_DURATION = 4.0  # Độ dài tối thiểu (giây)
MAX_DURATION = 8.0  # Độ dài tối đa (giây)
MAX_SILENCE = 1.5  # Ngắt câu nếu im lặng quá 1.5s

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(BASE_DIR, "bin")
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
TEMP_DIR = os.path.join(BASE_DIR, "temp_batch")
URLS_FILE = os.path.join(BASE_DIR, "urls.txt")

FFMPEG_EXE = os.path.join(BIN_DIR, "ffmpeg.exe")
YTDLP_EXE = os.path.join(BIN_DIR, "yt-dlp.exe")


def time_str_to_ms(t_str):
    t_str = t_str.replace(",", ".")
    try:
        h, m, s = t_str.split(":")
        s, ms = s.split(".")
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    except Exception:
        return 0


def ms_to_time_str(ms):
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_srt(srt_content):
    blocks = re.split(r"\n\n", srt_content.strip())
    parsed_data = []
    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            time_line = next((line for line in lines if "-->" in line), None)
            if time_line:
                start, end = time_line.split(" --> ")
                text_lines = [
                    line for line in lines if line != time_line and not line.isdigit()
                ]
                text = " ".join(text_lines).strip()
                if text:
                    parsed_data.append(
                        {
                            "start": start.strip().replace(",", "."),
                            "end": end.strip().replace(",", "."),
                            "text": text,
                        }
                    )
    return parsed_data


def remove_text_overlap(prev_text, curr_text):
    p_words = prev_text.lower().split()
    c_words = curr_text.lower().split()
    c_words_orig = curr_text.split()
    check_range = min(len(p_words), len(c_words))
    max_overlap = 0
    for k in range(check_range, 0, -1):
        if p_words[-k:] == c_words[:k]:
            max_overlap = k
            break
    if max_overlap > 0:
        return " ".join(c_words_orig[max_overlap:]).strip()
    return curr_text


def process_transcript_pipeline(raw_data):
    if not raw_data:
        return []

    step1 = [raw_data[0]]
    for i in range(1, len(raw_data)):
        prev = step1[-1]
        curr = raw_data[i]

        curr["text"] = remove_text_overlap(prev["text"], curr["text"])
        if not curr["text"]:
            continue

        p_end = time_str_to_ms(prev["end"])
        c_start = time_str_to_ms(curr["start"])
        c_end = time_str_to_ms(curr["end"])

        if c_start < p_end:
            c_start = p_end
            curr["start"] = ms_to_time_str(c_start)

        if (c_end - c_start) > 100:
            step1.append(curr)

    if not step1:
        return []
    final_merged = []
    curr_blk = step1[0].copy()
    curr_blk["start_ms"] = time_str_to_ms(curr_blk["start"])
    curr_blk["end_ms"] = time_str_to_ms(curr_blk["end"])

    for i in range(1, len(step1)):
        nxt = step1[i]
        nxt_s = time_str_to_ms(nxt["start"])
        nxt_e = time_str_to_ms(nxt["end"])

        curr_dur = (curr_blk["end_ms"] - curr_blk["start_ms"]) / 1000.0
        new_total = (nxt_e - curr_blk["start_ms"]) / 1000.0
        silence = (nxt_s - curr_blk["end_ms"]) / 1000.0

        if (
            (curr_dur < MIN_DURATION)
            and (new_total <= MAX_DURATION)
            and (silence <= MAX_SILENCE)
        ):
            curr_blk["end"] = nxt["end"]
            curr_blk["end_ms"] = nxt_e
            curr_blk["text"] += " " + nxt["text"]
        else:
            final_merged.append(curr_blk)
            curr_blk = nxt.copy()
            curr_blk["start_ms"] = nxt_s
            curr_blk["end_ms"] = nxt_e
    final_merged.append(curr_blk)
    return final_merged


def process_single_video(url, index):
    idx_str = f"{index:02d}"
    folder_name = f"Video_{idx_str}"

    temp_video_name = f"video_{idx_str}"
    temp_sub_name = f"sub_{idx_str}"

    transcript_filename = f"transcript_{idx_str}.txt"

    print(f"\n{'=' * 50}")
    print(f"[{folder_name}] Đang xử lý: {url}")

    video_output_dir = os.path.join(OUTPUT_ROOT, folder_name)
    if os.path.exists(video_output_dir):
        print(f" [i] Thư mục '{folder_name}' đã tồn tại. Reset...")
        shutil.rmtree(video_output_dir)
    os.makedirs(video_output_dir)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    print(f" [1/3] Downloading as {temp_video_name}...")

    cmd_dl = [
        YTDLP_EXE,
        "-x",
        "--audio-format",
        "wav",
        "--write-auto-sub",
        "--sub-lang",
        "vi",
        "--convert-subs",
        "srt",
        "-o",
        os.path.join(TEMP_DIR, f"{temp_video_name}.%(ext)s"),
        url,
    ]
    try:
        subprocess.run(
            cmd_dl, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(" [!] Lỗi tải video. Bỏ qua.")
        return

    downloaded_srt = next((f for f in os.listdir(TEMP_DIR) if f.endswith(".srt")), None)

    if not downloaded_srt:
        print(" [!] Không tìm thấy file sub.")
        return

    old_sub_path = os.path.join(TEMP_DIR, downloaded_srt)
    new_sub_path = os.path.join(TEMP_DIR, f"{temp_sub_name}.srt")
    os.rename(old_sub_path, new_sub_path)

    wav_path = os.path.join(TEMP_DIR, f"{temp_video_name}.wav")

    if not os.path.exists(wav_path):
        print(" [!] Không tìm thấy file audio.")
        return

    print(f" -> Temp files ready: {temp_video_name}.wav, {temp_sub_name}.srt")

    print(" [2/3] Processing Transcript...")
    with open(new_sub_path, "r", encoding="utf-8") as f:
        final_data = process_transcript_pipeline(parse_srt(f.read()))

    print(f" [3/3] Exporting to '{folder_name}/{transcript_filename}'...")

    manifest_path = os.path.join(video_output_dir, transcript_filename)

    with open(manifest_path, "w", encoding="utf-8") as f_out:
        for i, item in enumerate(final_data):
            file_name = f"{i:04d}.wav"
            out_path = os.path.join(video_output_dir, file_name)

            s_ms = time_str_to_ms(item["start"])
            e_ms = time_str_to_ms(item["end"])
            dur = (e_ms - s_ms) / 1000.0

            cmd_ffmpeg = [
                FFMPEG_EXE,
                "-y",
                "-i",
                wav_path,
                "-ss",
                item["start"],
                "-to",
                item["end"],
                "-vn",
                out_path,
            ]
            subprocess.run(
                cmd_ffmpeg, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            relative_path = f"{folder_name}/{file_name}"

            record = {
                "text": item["text"],
                "file": relative_path,
                "duration": round(dur, 4),
            }
            f_out.write(str(record) + ",\n")

    print(f" [OK] Hoàn tất: {folder_name}")


def main():
    if not os.path.exists(FFMPEG_EXE) or not os.path.exists(YTDLP_EXE):
        print("LỖI: Thiếu file .exe trong thư mục bin/")
        return

    if not os.path.exists(URLS_FILE):
        print(f"LỖI: Không tìm thấy file '{URLS_FILE}'.")
        return

    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Tìm thấy {len(urls)} links.")

    for i, url in enumerate(urls, 1):
        process_single_video(url, i)

    print("\n" + "=" * 50)
    print("ĐÃ XỬ LÝ XONG TOÀN BỘ!")


if __name__ == "__main__":
    main()
