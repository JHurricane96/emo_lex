import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--langs", type=str, nargs="+", required=True)
parser.add_argument("--align_dir", type=Path, required=True)
parser.add_argument("--exp_id", type=str, required=True)

parser.add_argument("--trans_dir", type=Path)
parser.add_argument("--emo_lex_dir", type=Path, required=True)

parser.add_argument("--nns_dir", type=Path, required=True)
parser.add_argument("--reports_dir", type=Path, required=True)

parser.add_argument("--skip_eval", action="store_true")

def create_expanded_path(path):
    path = Path(path)
    return path.expanduser()

def mkdir(path, parents=True, exist_ok=True):
    path.mkdir(parents=parents, exist_ok=exist_ok)

eval_align_script = "eval_align.py"
nns_script = "compute_nns.py"
eval_emos_script = "eval_emos.py"
create_emos_script = "create_emos.py"

if __name__ == "__main__":
    opt = parser.parse_args()
    report_file_loc = opt.reports_dir/opt.exp_id
    mkdir(report_file_loc)
    report_file = open(report_file_loc/"report.txt", "w")
    log_file = open(report_file_loc/"logs.txt", "w")
    for lang in opt.langs:
        print("Processing", lang)
        lang_align_dir = opt.align_dir/lang
        lang_code = lang.split("_")[0]
        src_emb_file = f"{lang_align_dir/lang}.vec"
        tgt_emb_file = f"{lang_align_dir}/eng.vec"
        trans_file = f"{opt.trans_dir/lang_code}_eng.txt"

        report_file.write(f"{lang}:\n")

        if not opt.skip_eval:
            subprocess.run(
                [
                    "python", "-u", f"{eval_align_script}",
                    "--src_emb", f"{src_emb_file}",
                    "--tgt_emb", f"{tgt_emb_file}",
                    "--dico_test", f"{trans_file}",
                    "--report_file", f"{report_file_loc}/eval_align_report.txt"
                ],
                stderr=subprocess.STDOUT,
                stdout=log_file,
                check=True)
            with open(report_file_loc/"eval_align_report.txt", "r") as fin:
                report_file.write(fin.read())
            print("Evaluated translation precision")
        
        nns_dir = opt.nns_dir/opt.exp_id
        mkdir(nns_dir)
        nns_file = f"{nns_dir/lang}.txt"
        nns_cmd = [
            "python", "-u", f"{nns_script}",
            "--src_emb", f"{src_emb_file}",
            "--tgt_emb", f"{tgt_emb_file}",
            "--nns_file", f"{nns_file}"
        ]
        if not opt.skip_eval:
            nns_cmd.extend(["--dico_test", f"{trans_file}"])
        subprocess.run(
            nns_cmd,
            stderr=subprocess.STDOUT,
            stdout=log_file,
            check=True
        )
        print("Calculated nearest neighbors")

        emos_dir = report_file_loc/"emos"
        mkdir(emos_dir)
        if not opt.skip_eval:
            emos_eval_dir = report_file_loc/"emos_eval"
            mkdir(emos_eval_dir)
            subprocess.run(
                [
                    "python", "-u", f"{eval_emos_script}",
                    "--trans_file", f"{nns_file}",
                    "--emo_lex", f"{opt.emo_lex_dir/lang_code}.txt",
                    "--report_file", f"{report_file_loc}/eval_emos_report.txt",
                    "--induct_emos_file", f"{emos_dir}/{lang}_emos.txt",
                    "--induct_emos_eval_file", f"{emos_eval_dir}/{lang}_emos.txt"
                ],
                stderr=subprocess.STDOUT,
                stdout=log_file,
                check=True
            )
            with open(report_file_loc/"eval_emos_report.txt", "r") as fin:
                report_file.write(fin.read())
            report_file.flush()
        else:
            subprocess.run(
                [
                    "python", "-u", f"{create_emos_script}",
                    "--trans_file", f"{nns_file}",
                    "--emo_lex", f"{opt.emo_lex_dir}/eng.txt",
                    "--induct_emos_file", f"{emos_dir}/{lang}_emos.txt",
                ],
                stderr=subprocess.STDOUT,
                stdout=log_file,
                check=True
            )

        print("Evaluated emotion correlations")

        report_file.write("\n")
        print("Finished", lang)
    report_file.close()
    log_file.close()
