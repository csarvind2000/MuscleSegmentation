"""Regenerate the single-column Springer/JIIM manuscript from the IEEEtran source,
so both stay in sync. Run after editing manuscript/manuscript.tex."""
import re, os
SRC = "/media/ranjhaa-local/volume23/sarcopenia/seg_benchmark/manuscript/manuscript.tex"
DST = "/media/ranjhaa-local/volume23/sarcopenia/seg_benchmark/manuscript_springer/manuscript.tex"

pre = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{url}
\usepackage{setspace}
\usepackage[hidelinks]{hyperref}
\graphicspath{{../figures/}}
\onehalfspacing
% keep floats near their text rather than piling at the end
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{5}
\begin{document}
"""
def convert(src, dst, is_supp=False):
    s = open(src).read().split(r"\begin{document}", 1)[1]
    s = pre + s
    if is_supp:
        # preserve the S1/S2... numbering that the stripped preamble carried
        renew = ("\\renewcommand{\\thesection}{S\\arabic{section}}\n"
                 "\\renewcommand{\\thetable}{S\\arabic{table}}\n"
                 "\\renewcommand{\\thefigure}{S\\arabic{figure}}\n")
        s = s.replace(r"\begin{document}", renew + r"\begin{document}", 1)
    s = s.replace(r"\IEEEoverridecommandlockouts", "")

    def _author(_m):
        return "\n".join([
            r"\author{Arvind Channarayapatna Srinivasa\textsuperscript{1}, Shamshekhar S.\ Patil\textsuperscript{2}\\[5pt]",
            r"\begin{minipage}{0.9\textwidth}\centering\small",
            r"\textsuperscript{1}Bioinformatics Institute (BII), Agency for Science, Technology and Research (A*STAR), 30 Biopolis Street, \#07-01 Matrix, 138671, Singapore\\[2pt]",
            r"\textsuperscript{2}Department of Computer Science, Dr.\ Ambedkar Institute of Technology, Visvesvaraya Technological University (VTU, Belagavi), Bengaluru, Karnataka 560056, India\\[3pt]",
            r"Emails: arvindcs@a-star.edu.sg (corresponding), shamshekhar.cs@drait.edu.in",
            r"\end{minipage}}",
            r"\date{}",
            r"\maketitle",
        ])
    s = re.sub(r"\\author\{.*?\}\n\n\\maketitle", _author, s, flags=re.S)
    s = s.replace(r"\author{}", r"\date{}")
    s = s.replace(r"\begin{IEEEkeywords}", r"\vspace{2mm}\noindent\textbf{Keywords:} ")
    s = s.replace(r"\end{IEEEkeywords}", "")
    # structured abstract: one labelled section per line (Springer style)
    for lab in ("Methods", "Results", "Conclusion"):
        s = s.replace("\\textbf{%s.}" % lab,
                      "\\par\\vspace{3pt}\\noindent\\textbf{%s.}" % lab)
    s = s.replace("figure*}", "figure}")
    s = s.replace(r"\begin{figure}[t]", r"\begin{figure}[htbp]")
    s = s.replace(r"0.92\textwidth", r"0.95\linewidth").replace(r"0.96\textwidth", r"0.95\linewidth")
    s = s.replace(r"\columnwidth", r"\linewidth")
    s = s.replace(r"width=0.8\linewidth]{dataset_overview.png}", r"width=0.55\linewidth]{dataset_overview.png}")
    s = s.replace(r"width=0.72\linewidth]{dataset_external.png}", r"width=0.5\linewidth]{dataset_external.png}")
    # qual_methods.png stays at 0.98\textwidth = full page width in single-column (6 panels)
    s = s.replace(r"width=0.80\textwidth]{qual_factors.png}", r"width=0.82\textwidth]{qual_factors.png}")
    s = s.replace(r"width=0.98\linewidth]{qual_thigh.png}", r"width=0.85\linewidth]{qual_thigh.png}")
    s = s.replace(r"width=0.95\linewidth]{qual_aattct.png}", r"width=0.72\linewidth]{qual_aattct.png}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(s)
    print("wrote", dst)


convert(SRC, DST)
SUP = "/media/ranjhaa-local/volume23/sarcopenia/seg_benchmark/manuscript/supplementary.tex"
SUP_DST = "/media/ranjhaa-local/volume23/sarcopenia/seg_benchmark/manuscript_springer/supplementary.tex"
if os.path.exists(SUP):
    convert(SUP, SUP_DST, is_supp=True)
