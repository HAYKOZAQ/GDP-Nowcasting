from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
PART_FILES = [
    ROOT / "1st_part.tex",
    ROOT / "2nd_part.tex",
    ROOT / "3rd_part.tex",
]
OUTPUT_FILE = ROOT / "main.tex"

META = {
    "university_hy": "ԵՐԵՎԱՆԻ ՊԵՏԱԿԱՆ ՀԱՄԱԼՍԱՐԱՆ",
    "faculty_hy": "ՄԱԹԵՄԱՏԻԿԱՅԻ ԵՎ ՄԵԽԱՆԻԿԱՅԻ ՖԱԿՈՒԼՏԵՏ",
    "department_hy": "ՀԱՎԱՆԱԿԱՆՈՒԹՅՈՒՆՆԵՐԻ ՏԵՍՈՒԹՅԱՆ ԵՎ ՄԱԹԵՄԱՏԻԿԱԿԱՆ ՎԻՃԱԿԱԳՐՈՒԹՅԱՆ ԱՄԲԻՈՆ",
    "program_hy": "ԿԻՐԱՌԱԿԱՆ ՎԻՃԱԿԱԳՐՈՒԹՅՈՒՆ ԵՎ ՏՎՅԱԼՆԵՐԻ ԳԻՏՈՒԹՅՈՒՆ ԿՐԹԱԿԱՆ ԾՐԱԳԻՐ",
    "university_en": "YEREVAN STATE UNIVERSITY",
    "faculty_en": "FACULTY OF MATHEMATICS AND MECHANICS",
    "department_en": "DEPARTMENT OF PROBABILITY THEORY AND MATHEMATICAL STATISTICS",
    "program_en": "APPLIED STATISTICS AND DATA SCIENCE MASTER'S PROGRAM",
    "author_hy": "ԶԱՔԱՐՅԱՆ ՀԱՅԿ ԱՐՄԵՆԻ",
    "author_en": "ZAKARYAN HAYK ARMEN",
    "supervisor_hy": "Հարությունյան Արշալույս",
    "supervisor_en": "Arshaluys Harutyunyan",
    "program_head_hy": "ֆ.մ.գ.դ., ասիստենտ Կարեն Քեռյան",
    "program_head_en": "Karen Keryan",
    "title_hy": "ՀՆԱ աճի կանխատեսում՝ օգտագործելով այլընտրանքային տվյալների աղբյուրներ",
    "title_ru": "Прогнозирование роста ВВП с использованием альтернативных источников данных",
    "title_en": "Nowcasting GDP Growth Using Alternative Data Sources",
    "degree_line_hy": "«Վիճակագրություն» մասնագիտությամբ առկա մագիստրոսի որակավորման աստիճանի հայցման համար",
    "degree_line_en": "Submitted in Partial Fulfillment of the Requirements for the Full-time Degree of Master of Statistics",
    "city_hy": "ԵՐԵՎԱՆ",
    "city_en": "YEREVAN",
    "year": "2026",
}

PREAMBLE = dedent(
    r"""
    \documentclass[12pt,a4paper]{report}

    \usepackage{iftex}
    \ifLuaTeX
      \usepackage{fontspec}
      \usepackage{polyglossia}
      \setmainlanguage{english}
      \setotherlanguages{armenian,russian}
      \defaultfontfeatures{Ligatures=TeX}
      \setmainfont{Times New Roman}
      \setsansfont{Arial}
      \newfontfamily\armenianfont[Script=Armenian]{Sylfaen}
      \newfontfamily\armenianfontsf[Script=Armenian]{Sylfaen}
      \newfontfamily\armenianfonttt[Script=Armenian]{Sylfaen}
      \newfontfamily\armenianfrontfont[Script=Armenian]{Arial}
      \newfontfamily\russianfont{Times New Roman}
    \else
      \errmessage{This combined thesis file must be compiled with LuaLaTeX}
    \fi

    \usepackage{graphicx}
    \usepackage{geometry}
    \usepackage{setspace}
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{bm}
    \usepackage[numbers]{natbib}
    \usepackage[hidelinks]{hyperref}
    \usepackage{xurl}
    \usepackage{caption}
    \usepackage{tocloft}
    \usepackage{float}
    \usepackage{booktabs}
    \usepackage{longtable}
    \usepackage{tabularx}
    \usepackage{array}
    \usepackage{microtype}

    \geometry{
      a4paper,
      left=30mm,
      right=15mm,
      top=20mm,
      bottom=25mm
    }
    \setlength{\parindent}{1cm}

    \onehalfspacing
    \hypersetup{
      hidelinks,
      colorlinks=false,
      pdfborder={0 0 0},
      linkbordercolor={1 1 1},
      citebordercolor={1 1 1},
      urlbordercolor={1 1 1},
      filebordercolor={1 1 1}
    }
    \graphicspath{{../}{../figures/}{../results/figures/}{../results/forecasts/}{figures/}}
    \newcolumntype{Y}{>{\raggedright\arraybackslash}X}
    \renewcommand{\arraystretch}{1.12}
    \renewcommand{\contentsname}{CONTENTS}
    \setlength{\cftbeforetoctitleskip}{-0.35cm}
    \setlength{\cftaftertoctitleskip}{0.45cm}
    \setlength{\cftbeforechapskip}{2pt}

    \usepackage{titlesec}
    \titleformat{\chapter}[block]
      {\normalfont\Large\bfseries\centering}
      {\thechapter.}
      {1em}
      {\MakeUppercase}

    \begin{document}
    """
)


def extract_body(text: str) -> str:
    match = re.search(r"\\begin\{document\}(?P<body>.*)\\end\{document\}", text, flags=re.S)
    if not match:
        raise ValueError("Could not find document body.")
    return match.group("body").strip()


def strip_title_commands(body: str) -> str:
    body = re.sub(r"^\s*\\maketitle\s*", "", body, count=1, flags=re.S)
    body = re.sub(r"^\s*\\tableofcontents\s*", "", body, count=1, flags=re.S)
    return body.strip()


def extract_abstract(body: str) -> tuple[str, str]:
    match = re.search(r"\\chapter\*\{Abstract\}(?P<abstract>.*?)(?=\\chapter\{)", body, flags=re.S)
    if not match:
        return "", body
    abstract = match.group("abstract").strip()
    body = body[: match.start()] + body[match.end() :]
    return abstract, body.strip()


def extract_bibliography_items(body: str) -> tuple[list[str], str]:
    bib_match = re.search(
        r"(?P<full>(\\chapter\*\{References\}.*?\\addcontentsline\{toc\}\{chapter\}\{References\}\s*)?"
        r"\\begin\{thebibliography\}\{[^}]*\}(?P<items>.*?)\\end\{thebibliography\})",
        body,
        flags=re.S,
    )
    if not bib_match:
        bib_match = re.search(
            r"(?P<full>\\begin\{thebibliography\}\{[^}]*\}(?P<items>.*?)\\end\{thebibliography\})",
            body,
            flags=re.S,
        )
    if not bib_match:
        return [], body

    items_block = bib_match.group("items").strip()
    item_matches = re.findall(
        r"(\\bibitem\{[^}]+\}.*?)(?=(\\bibitem\{[^}]+\})|\Z)",
        items_block,
        flags=re.S,
    )
    items = [item.strip() for item, _ in item_matches]
    body_without_bib = (body[: bib_match.start()] + body[bib_match.end() :]).strip()
    return items, body_without_bib


def strip_part2_ending(body: str) -> str:
    marker = r"\chapter{Limitations and Directions for Further Research}"
    if marker in body:
        body = body.split(marker, 1)[0].rstrip()
    return body


def extract_part3_final_conclusion(body: str) -> tuple[str, str]:
    marker = r"\subsection{Final Conclusion for the Thesis}"
    if marker in body:
        before, after = body.split(marker, 1)
        return after.strip(), before.rstrip()
    marker = r"\section{Final Conclusion for the Thesis}"
    if marker not in body:
        return "", body
    before, after = body.split(marker, 1)
    return after.strip(), before.rstrip()


def transform_part3(body: str) -> str:
    title = r"\section{Nowcasting Framework, Model Construction, and Empirical Results}"
    if title in body:
        body = body.replace(title, "", 1).strip()
    return "\\chapter{Nowcasting Framework, Model Construction, and Empirical Results}\n\n" + body


def normalize_spacing(block: str) -> str:
    block = re.sub(r"\n{3,}", "\n\n", block.strip())
    return block + "\n"


def build_bibliography(items: list[str]) -> str:
    unique_items: list[str] = []
    seen_keys: set[str] = set()

    for item in items:
        key_match = re.match(r"\\bibitem\{([^}]+)\}", item)
        if not key_match:
            continue
        key = key_match.group(1)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_items.append(item)

    if not unique_items:
        return ""

    # Sort bibliography alphabetically by the reference tag or author
    def sort_key(item: str) -> str:
        match = re.search(r"\\bibitem\{[^}]+\}\s*(.*?)(?=\\href|\Z)", item, flags=re.S)
        if match:
            return match.group(1).strip().lower()
        return item.lower()

    unique_items.sort(key=sort_key)

    joined = "\n\n".join(unique_items)
    return (
        "\n\\renewcommand{\\bibname}{References}\n"
        "\\cleardoublepage\n"
        "\\phantomsection\n"
        "\\addcontentsline{toc}{chapter}{References}\n"
        "\\begin{thebibliography}{99}\n\n"
        f"{joined}\n\n"
        "\\end{thebibliography}\n"
    )


def build_front_matter(abstract_text: str) -> str:
    armenian_summary = (
        "Աշխատությունը նվիրված է Հայաստանի ընթացիկ եռամսյակի ՀՆԱ աճի գնահատմանը մինչև պաշտոնական ազգային "
        "հաշիվների հրապարակումը։ Թեզում համադրվում են ավանդական մակրոտնտեսական ցուցանիշները և այլընտրանքային "
        "տվյալների աղբյուրները՝ փոխարժեքների, ապրանքային գների, որոնողական ակտիվության, տրանսֆերտների, "
        "զբոսաշրջության և ամսական ոլորտային ցուցանիշների տեսքով։ Այս մոտեցումը կարևոր է Հայաստանի նման փոքր "
        "և արտաքին շոկերի նկատմամբ զգայուն տնտեսության համար, որտեղ շրջադարձային փոփոխությունները հաճախ առաջինը "
        "երևում են բարձր հաճախականության տվյալներում։\n\n"
        "Աշխատության շրջանակում կառուցվել է փուլային nowcasting համակարգ, որը տարբերակում է Early, Mid և Late "
        "տեղեկատվական փուլերը։ Ցույց է տրվում, որ այլընտրանքային տվյալների նշանակությունը հատկապես մեծ է "
        "եռամսյակի սկզբում, երբ պաշտոնական նույն եռամսյակի տեղեկությունը սահմանափակ է։ Տեղեկատվական դաշտի "
        "հարստացման հետ առավել արդյունավետ են դառնում ադապտիվ համակցված մոդելները, իսկ շոկային ճշգրտմամբ "
        "տարբերակները բարելավում են արդյունքները ճգնաժամային եռամսյակներում։\n\n"
        "Արդյունքները ցույց են տալիս, որ Հայաստանը 2025 թվական է մտել հարաբերականորեն կայուն աճի իներցիայով, "
        "սակայն աճի կառուցվածքը դարձել է ավելի անհավասարաչափ։ Շինարարությունը և անշարժ գույքի հետ կապված "
        "ակտիվությունը պահպանել են առանցքային դերը, ՏՀՏ ոլորտը շարունակել է ունենալ ռազմավարական նշանակություն, "
        "իսկ հարկաբյուջետային և ֆինանսական կենտրոնացման ռիսկերը դարձել են ավելի տեսանելի։ Ընդհանուր "
        "եզրակացությունն այն է, որ կարճաժամկետ բարենպաստ հեռանկարը կայուն կմնա միայն այն դեպքում, եթե արագ "
        "հատվածային աճը ուղեկցվի արտադրողականության բարձրացմամբ, կարգապահ մակրոտնտեսական քաղաքականությամբ և "
        "պահանջարկի կառուցվածքային վերաբալանսավորմամբ։"
    )

    return dedent(
        f"""
        \\hypersetup{{pageanchor=false}}

        \\begin{{titlepage}}
        \\begin{{center}}
        {{\\armenianfrontfont\\bfseries\\Large \\MakeUppercase{{{META["university_hy"]}}}\\par}}
        \\vspace{{0.6cm}}
        {{\\armenianfrontfont\\bfseries\\Large \\MakeUppercase{{{META["faculty_hy"]}}}\\par}}
        \\vspace{{1.4cm}}
        {{\\armenianfrontfont\\bfseries\\Large {META["department_hy"]}\\par}}
        \\vspace{{1.8cm}}
        {{\\armenianfrontfont\\bfseries\\Large \\MakeUppercase{{{META["program_hy"]}}}\\par}}
        \\vspace{{2.1cm}}
        {{\\armenianfrontfont\\bfseries\\Large {META["author_hy"]}\\par}}
        \\vspace{{1.5cm}}
        {{\\armenianfrontfont\\bfseries\\LARGE ՄԱԳԻՍՏՐՈՍԱԿԱՆ ԹԵԶ\\par}}
        \\vspace{{1.5cm}}
        {{\\armenianfrontfont\\bfseries\\Large {META["title_hy"]}\\par}}
        \\vspace{{1.4cm}}
        {{\\armenianfrontfont\\bfseries\\large {META["degree_line_hy"]}\\par}}
        \\vfill
        {{\\armenianfrontfont\\bfseries\\Large {META["city_hy"]} {META["year"]}\\par}}
        \\end{{center}}
        \\end{{titlepage}}

        \\begin{{titlepage}}
        \\begin{{center}}
        {{\\sffamily\\bfseries\\Large {META["university_en"]}\\par}}
        \\vspace{{0.6cm}}
        {{\\sffamily\\bfseries\\Large {META["faculty_en"]}\\par}}
        \\vspace{{1.4cm}}
        {{\\sffamily\\bfseries\\Large {META["department_en"]}\\par}}
        \\vspace{{1.8cm}}
        {{\\sffamily\\bfseries\\Large {META["program_en"]}\\par}}
        \\vspace{{2.1cm}}
        {{\\sffamily\\bfseries\\Large {META["author_en"]}\\par}}
        \\vspace{{1.5cm}}
        {{\\sffamily\\bfseries\\LARGE MASTER'S THESIS\\par}}
        \\vspace{{1.5cm}}
        {{\\sffamily\\bfseries\\Large {META["title_en"]}\\par}}
        \\vspace{{1.4cm}}
        {{\\sffamily\\bfseries\\large {META["degree_line_en"]}\\par}}
        \\vfill
        {{\\sffamily\\bfseries\\Large {META["city_en"]} {META["year"]}\\par}}
        \\end{{center}}
        \\end{{titlepage}}

        \\clearpage
        \\thispagestyle{{empty}}
        \\vspace*{{1.3cm}}

        \\noindent {{\\armenianfrontfont\\bfseries Ուսանող՝}}\\hfill
        \\begin{{tabular}}[t]{{c}}
        \\rule{{9.0cm}}{{0.4pt}} \\\\[-0.1cm]
        \\footnotesize\\textarmenian{{ստորագրություն}}
        \\end{{tabular}} \\vspace{{0.8cm}}

        \\begin{{center}}
        \\begin{{tabular}}{{c}}
        {{\\armenianfrontfont\\bfseries Զաքարյան Հայկ}} \\\\[-0.1cm]
        \\rule{{15.0cm}}{{0.4pt}} \\\\[-0.1cm]
        \\footnotesize\\textarmenian{{ազգանուն, անուն}}
        \\end{{tabular}}
        \\end{{center}} \\vspace{{1.2cm}}

        \\noindent {{\\armenianfrontfont\\bfseries Գիտական ղեկավար՝}}\\hfill
        \\begin{{tabular}}[t]{{c}}
        \\rule{{9.0cm}}{{0.4pt}} \\\\[-0.1cm]
        \\footnotesize\\textarmenian{{ստորագրություն}}
        \\end{{tabular}} \\vspace{{0.8cm}}

        \\begin{{center}}
        \\begin{{tabular}}{{c}}
        {{\\armenianfrontfont\\bfseries Դասախոս, {META["supervisor_hy"]}}} \\\\[-0.1cm]
        \\rule{{15.0cm}}{{0.4pt}} \\\\[-0.1cm]
        \\footnotesize\\textarmenian{{գիտ. աստիճան, կոչում, ազգանուն, անուն}}
        \\end{{tabular}}
        \\end{{center}} \\vspace{{1.0cm}}

        \\vspace{{2.2cm}}
        \\noindent {{\\armenianfrontfont\\bfseries «Թույլատրել պաշտպանության»}}\\\\[1.3cm]

        \\noindent {{\\armenianfrontfont\\bfseries Մագիստրոսական կրթական ծրագրի ղեկավար՝}}\\\\[0.2cm]
        \\begin{{flushright}}
        {{\\armenianfrontfont\\bfseries {META["program_head_hy"]}}}
        \\end{{flushright}}

        \\vfill
        \\noindent \\textarmenian{{<<\\hspace{{1cm}}>> մայիսի {META["year"]}թ.}}

        \\clearpage
        \\setcounter{{page}}{{4}}
        \\phantomsection
        \\addcontentsline{{toc}}{{chapter}}{{Համառոտագիր}}
        \\vspace*{{0.2cm}}
        {{\\armenianfrontfont\\bfseries Թեզի վերնագիրը՝}}\\\\[0.75cm]
        \\noindent {{\\armenianfrontfont\\bfseries Հայերեն՝}} \\textarmenian{{{META["title_hy"]}}},\\\\[0.35cm]
        \\noindent {{\\armenianfrontfont\\bfseries Ռուսերեն՝}} \\textrussian{{{META["title_ru"]}}},\\\\[0.35cm]
        \\noindent {{\\armenianfrontfont\\bfseries Անգլերեն՝}} {META["title_en"]}\\\\[1.05cm]

        \\begin{{center}}
        {{\\armenianfrontfont\\bfseries Համառոտագիր}}
        \\end{{center}}
        \\begin{{armenian}}
        \\begin{{spacing}}{{0.98}}
        \\small
        \\sloppy
        {armenian_summary}
        \\par
        \\end{{spacing}}
        \\end{{armenian}}
        \\clearpage
        \\phantomsection
        \\addcontentsline{{toc}}{{chapter}}{{Abstract}}
        \\vspace*{{0.2cm}}

        \\begin{{center}}
        {{\\sffamily\\bfseries Abstract}}
        \\end{{center}}
        \\begin{{english}}
        \\begin{{spacing}}{{0.98}}
        \\small
        \\sloppy
        {abstract_text}

        This topic is of critical importance for policymakers and econometricians, as it provides a robust framework to generate timely, high-frequency signals of economic activity before official national accounts are published, enabling more responsive, proactive, and data-driven policy decisions.
        \\par
        \\end{{spacing}}
        \\end{{english}}
        \\clearpage
        \\tableofcontents

        \\clearpage
        \\hypersetup{{pageanchor=true}}
        """
    )


def build_final_conclusion() -> str:
    return dedent(
        r"""
        \chapter{Conclusion}

        This thesis examined Armenia's recent growth dynamics from two linked perspectives. The first was a high-frequency macroeconomic reading of the economy through external, real-sector, labor-market, fiscal, and financial indicators. The second was a pseudo real-time nowcasting framework designed to estimate current-quarter GDP growth before the publication of official national accounts. Taken together, these two parts answer the same broader question: whether Armenia's post-2022 expansion should be interpreted as a durable structural shift or as a more temporary configuration supported by favorable but potentially unstable conditions.

        The empirical evidence supports a balanced conclusion. Armenia remained resilient through 2026 Q1, but that resilience was uneven in composition. Construction, urban real estate, selected services, and ICT remained central drivers of activity. Labor-market conditions stayed relatively tight, inflation was contained, and the external environment continued to provide support through remittances, tourism, and service exports. At the same time, the analysis repeatedly showed concentration risk: growth was not broad-based enough to justify an unqualified structural-boom interpretation, and several of the strongest near-term supports remained vulnerable to external normalization, financial concentration, and demand rebalancing.

        The nowcasting chapter strengthens that broader macroeconomic interpretation. It shows that Armenian GDP can be tracked credibly in real time, but only if model choice reflects the information stage and the possibility of abrupt regime change. In the refreshed benchmark run, \texttt{StackingNowcast} is the best operational specification across all stages (Early 2.382\%, Mid 2.369\%, Late 1.875\%, average 2.209\%). At the same time, this does not remove the need for targeted crisis repair: \texttt{EarlyShockAdjusted} remains the relevant benchmark for the hardest month-1 collapse case (2020 Q2 Early), while \texttt{DFM} remains the most theory-grounded structural benchmark.

        One of the central contributions of the thesis is therefore methodological as well as substantive. For Armenia, alternative data sources are not decorative additions to a conventional macro framework. They are most valuable precisely where the real-time policy problem is hardest: at the start of the quarter, when the official information set is incomplete and the economy may already be changing direction. The evidence suggests that fast market variables do most of the heavy lifting in that setting, while the Google Trends composite block adds only a modest and statistically weak incremental gain. The newer administrative fintech indicators, such as payment-card and e-money proxies, are more useful as structural context than as a dramatic standalone breakthrough. Exchange-rate movements, commodity prices, search intensity, remittance dynamics, tourism flows, payment behavior, and fast monthly sector indicators therefore improve the timing and realism of the nowcast when conventional quarterly GDP is still unavailable, but not all alternative sources contribute equally.

        The main policy implication follows directly from this combined evidence. Armenia's short-term outlook should be viewed as favorable but conditional. A constructive baseline is justified, yet it should be paired with caution about concentration in construction-led demand, macro-financial exposure, and the dependence of headline growth on a relatively narrow set of channels. The forward projection through 2026 Q4 points to moderate deceleration rather than collapse, with point forecasts of 105.444 for 2026 Q2, 104.446 for 2026 Q3, and 104.206 for 2026 Q4. In that sense, the most important result of the thesis is not a single forecast number. It is the demonstration that timely, stage-aware, and shock-sensitive monitoring can distinguish between strong growth, imbalanced growth, and genuinely durable structural strengthening.

        Overall, the thesis concludes that Armenia's economy during the period studied was neither fragile in a simple sense nor fully transformed. It was undergoing real-time restructuring under unusually complex conditions. A framework that combines conventional indicators with alternative data sources is therefore essential both for practical nowcasting and for a more credible interpretation of what current GDP growth actually means.
        """
    ).strip() + "\n"


def main() -> None:
    raw_parts = [path.read_text(encoding="utf-8") for path in PART_FILES]
    part_bodies = [strip_title_commands(extract_body(text)) for text in raw_parts]

    # Collect bibliography entries from the full source parts first, so trimming
    # conclusions or appendices does not accidentally drop references.
    bib_items: list[str] = []
    for body in part_bodies:
        bib_part, _ = extract_bibliography_items(body)
        bib_items.extend(bib_part)

    abstract_block, part2_body = extract_abstract(part_bodies[1])
    part2_body = strip_part2_ending(part2_body)
    _, part3_body = extract_part3_final_conclusion(part_bodies[2])
    processed_parts: list[str] = []
    bodies_for_merge = [
        part2_body,
        transform_part3(part3_body),
    ]

    for body in bodies_for_merge:
        _, clean_body = extract_bibliography_items(body)
        processed_parts.append(normalize_spacing(clean_body))

    chunks = [PREAMBLE, build_front_matter(normalize_spacing(abstract_block).strip())]
    chunks.extend(processed_parts)
    chunks.append(build_final_conclusion())

    bibliography = build_bibliography(bib_items)
    if bibliography:
        chunks.append(bibliography)

    chunks.append("\n\\end{document}\n")
    OUTPUT_FILE.write_text("".join(chunks), encoding="utf-8")
    print(f"Generated {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
