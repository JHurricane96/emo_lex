import argparse
import asyncio
from pathlib import Path
import math
import shutil
from device_set import DeviceSet

parser = argparse.ArgumentParser()

parser.add_argument("--langs", type=str, nargs="+", required=True)
parser.add_argument("--bible_dir", type=Path, required=True)
parser.add_argument("--align_dir", type=Path, required=True)
parser.add_argument("--emb_dir", type=Path, required=True)
parser.add_argument("--overwrite_align", action="store_true")
parser.add_argument("--create_emb", action="store_true")
parser.add_argument("--num_gpus", type=int, required=True)

parser.add_argument("--algorithm", choices=["fb", "nlm", "vecmap"], required=True)

parser.add_argument("--vecmap_dir", type=Path)
parser.add_argument("--nlm_dir", type=Path)
parser.add_argument("--fasttext_dir", type=Path)

parser.add_argument("--sid_bible_dir", type=Path)
parser.add_argument("--eng_emb_file")

parser.add_argument("--nlm_preproc_dir", type=Path)
parser.add_argument("--nlm_preprocess", action="store_true")
parser.add_argument("--nlm_modified", action="store_true")

def file_linecount(file_name):
    with open(file_name, "r") as f:
        for i, _ in enumerate(f):
            pass
    return i

async def align_emb_vecmap(lang, device_set, opt):
    lang_align_dir = (opt.align_dir/lang)
    aligned_emb_file = lang_align_dir/f"{lang}.vec"
    if aligned_emb_file.exists() and not opt.overwrite_align:
        print(f"Skip aligning {lang} as already done")
        return
    
    preproc_dir = opt.emb_dir

    lang_align_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)

    if lang.endswith("_sm"):
        freq = 2
        # lang_id = lang + "_eng"
        # eng_id = "eng_" + lang
        lang_id = lang
        eng_id = "eng"
    else:
        freq = 5
        lang_id = lang
        eng_id = "eng"

    preproc_log_file = lang_align_dir/"logs_preproc.txt"
    align_log_file = lang_align_dir/"logs_align.txt"

    device_id = await device_set.acquire_device()

    if opt.create_emb or not (preproc_dir/f"{lang_id}.vec").is_file():
        p = await asyncio.create_subprocess_shell(
            f"{opt.fasttext_dir}/fasttext skipgram "
            f"-input {opt.bible_dir/lang_id}.txt "
            f"-output {preproc_dir/lang_id} "
            f"-epoch 25 "
            f"-lr 0.1 "
            f"-thread 6 "
            f"-dim 100 "
            f"-minCount {freq} "
            f">> {preproc_log_file} 2>&1 "
        )
        await p.wait()
        if p.returncode != 0:
            print(f"Error preprocessing {lang}")

        (preproc_dir/f"{lang_id}.bin").unlink()

        if lang.endswith("_sm") and (opt.create_emb or not (preproc_dir/f"{eng_id}.vec").is_file()):
            p = await asyncio.create_subprocess_shell(
                f"{opt.fasttext_dir}/fasttext skipgram "
                f"-input {opt.bible_dir/eng_id}.txt "
                f"-output {preproc_dir/eng_id} "
                f"-epoch 25 "
                f"-lr 0.05 "
                f"-thread 6 "
                f"-dim 100 "
                f"-minCount 2 "
                f">> {preproc_log_file} 2>&1 "
            )
            await p.wait()
        print(f"Preprocessed {lang}")
    else:
        print(f"Skip preprocessing {lang} as already done")

    eng_emb_file = opt.eng_emb_file if opt.eng_emb_file is not None else f"{preproc_dir/eng_id}.vec"
    sid_bible_cmd_str = ""
    if opt.sid_bible_dir:
        sid_bible_cmd_str = (
            f"--src_txt_file {opt.sid_bible_dir/lang}.txt "
            f"--tgt_txt_file {opt.sid_bible_dir}/eng.txt "
        )
    p = await asyncio.create_subprocess_shell(
        f"python -u {opt.vecmap_dir}/map_embeddings.py "
        f"--unsupervised "
        f"{preproc_dir/lang_id}.vec {eng_emb_file} "
        f"{lang_align_dir/lang}.vec {lang_align_dir}/eng.vec "
        f"{sid_bible_cmd_str}"
        f"--device {device_id} "
        f"--cuda -v "
        f">> {align_log_file} 2>&1 "
    )
    await p.wait()
    if p.returncode != 0:
        print(f"Error aligning {lang}")

    await device_set.release_device(device_id)
    print(f"Aligned {lang}")

async def align_emb_fb(lang, device_set, opt):
    lang_align_dir = (opt.align_dir/lang)
    aligned_emb_file = lang_align_dir/f"{lang}.vec"
    if aligned_emb_file.exists() and not opt.overwrite_align:
        print(f"Skip aligning {lang} as already done")
        return

    preproc_dir = opt.emb_dir

    lang_align_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)

    freq = 2 if lang.endswith("_sm") else 5

    preproc_log_file = lang_align_dir/"logs_preproc.txt"
    align_log_file = lang_align_dir/"logs_align.txt"
    lang_code = lang.split("_")[0]

    device_id = await device_set.acquire_device()

    if opt.create_emb or not (preproc_dir/f"{lang}.vec").is_file():
        p = await asyncio.create_subprocess_shell(
            f"{opt.fasttext_dir}/fasttext skipgram "
            f"-input {opt.bible_dir/lang}.txt "
            f"-output {preproc_dir/lang} "
            f"-epoch 25 "
            f"-lr 0.1 "
            f"-thread 6 "
            f"-dim 100 "
            f"-minCount {freq} "
            f">> {preproc_log_file} 2>&1 "
        )
        await p.wait()
        print(f"Preprocessed {lang}")
    else:
        print(f"Skip preprocessing {lang} as already done")

    non_en_lc = file_linecount(f"{preproc_dir/lang}.vec")
    en_lc = file_linecount(preproc_dir/"eng.vec")
    vocab_size = min(en_lc, non_en_lc)
    n_epoch = 5
    batch_size = math.floor(vocab_size / (2 ** (n_epoch - 1)))
    learning_rate = batch_size/2

    p = await asyncio.create_subprocess_shell(
        f"python -u {opt.fasttext_dir}/alignment/unsup_align.py "
        f"--model_src {preproc_dir/lang}.vec "
        f"--model_tgt {preproc_dir}/eng.vec "
        f"--output_src {lang_align_dir/lang}.vec "
        f"--output_tgt {lang_align_dir}/eng.vec "
        f"--nepoch {n_epoch} "
        f"--bsz {batch_size} "
        f"--lr {learning_rate} "
        f"--nmax {vocab_size} "
        f"--device {device_id} "
        f">> {align_log_file} 2>&1 "
    )
    await p.wait()
    await device_set.release_device(device_id)
    print(f"Aligned {lang}")

async def align_emb_nlm(lang, device_set, opt):
    lang_align_dir = (opt.align_dir/lang)
    aligned_emb_file = lang_align_dir/f"{lang}.vec"
    if aligned_emb_file.exists():
        return

    preproc_dir = opt.nlm_preproc_dir

    lang_align_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)

    freq = 2 if lang.endswith("_sm") else 5

    preproc_log_file = lang_align_dir/"logs_preproc.txt"
    align_log_file = lang_align_dir/"logs_align.txt"

    if opt.nlm_preprocess:
        if opt.nlm_modified:
            preproc_dir_cmd_str = f"-save_dir {preproc_dir}/ "
        else:
            preproc_dir_cmd_str = ""
        p = await asyncio.create_subprocess_shell(
            f"python {opt.nlm_dir/'preprocess.py'} "
            f"-train {opt.bible_dir/lang}.txt {opt.bible_dir}/eng.txt "
            f"-V_min_freq {freq} 3 "
            f"-save_name {lang} "
            f"-output_vocab "
            f"{preproc_dir_cmd_str}"
            f">> {preproc_log_file} 2>&1 "
        )
        await p.wait()
        print(f"Preprocessed {lang}")
    else:
        print(f"Skip preprocessing {lang} as already done")

    device_id = await device_set.acquire_device()
    if opt.nlm_modified:
        opts_cmd_str = \
        f"-data_dir {preproc_dir}/ "
        f"-learning_rate 0.2 "
        f"-stop_threshold 1.0 "
    else:
        opts_cmd_str = \
        f"-learning_rate 1.0 "
        f"-stop_threshold 0.99 "
    p = await asyncio.create_subprocess_shell(
        f"python -u {opt.nlm_dir}/train.py "
        f"-data {lang} "
        f"-gpuid {device_id} "
        f"-save_dir {lang_align_dir} "
        f"-batch_size 64 "
        f"-epoch_size 20 "
        f"-opt_type SGD "
        f"-n_layer 2 "
        f"-emb_size 300 "
        f"-h_size 300 "
        f"-seed 111 "
        f"-dr_rate 0.3 "
        f"-remove_models "
        f"{opts_cmd_str}"
        f">> {align_log_file} 2>&1 "
    )
    await p.wait()
    await device_set.release_device(device_id)
    shutil.move(lang_align_dir/f"{lang}.lang0.vec", aligned_emb_file)
    shutil.move(lang_align_dir/f"{lang}.lang1.vec", lang_align_dir/"eng.vec")
    print(f"Aligned {lang}")

async def main():
    opt = parser.parse_args()
    device_set = DeviceSet(opt.num_gpus)
    task_functions = {"fb": align_emb_fb, "nlm": align_emb_nlm, "vecmap": align_emb_vecmap}
    task_function = task_functions.get(opt.algorithm)
    if task_function == None:
        print("Could not retrieve task function")
        return
    align_tasks = [asyncio.create_task(task_function(lang, device_set, opt)) for lang in opt.langs]
    await asyncio.gather(*align_tasks)

if __name__ == "__main__":
    asyncio.run(main())
