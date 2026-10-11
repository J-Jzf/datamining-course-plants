"""A local image index containing only figures generated in the current run."""
from html import escape
from pathlib import Path

LABELS = {
    '01_region_clusters': '区域聚类地图', '02_recorded_richness': '记录物种数地图',
    '03_stable_rule_geography': '稳定关联规则的地理分布',
    '04_region_similarity_dendrogram': '层次树与地区相似度',
    '05_floristic_embedding': '植物组成的二维嵌入',
    '06_rule_support_confidence': '规则支持度、置信度与提升度',
    '07_rule_heldout_validation': '训练与留出集规则指标',
    '08_bootstrap_stability': 'Bootstrap 聚类稳定性',
    '09_rule_stability': '规则重现频率与指标区间',
    '10_algorithm_efficiency': 'Apriori 与 FP-growth 效率',
    '11_genus_validation': '按属划分验证',
    '12_broad_species_sensitivity': '广布植物敏感性',
    '13_metric_sensitivity': '距离、连接及支持度敏感性',
    '13_metric_support_sensitivity': '距离、连接及支持度敏感性',
    '14_spatial_relation': '地理与植物组成距离',
    '15_support_sensitivity': '支持度敏感性',
    '16_masked_region_completion': '隐藏地区的记录补全',
    'method_framework': '方法 Framework', 'plant_topic_overview': '研究主题介绍',
}


def write_gallery(paths, output_dir):
    output_dir = Path(output_dir).resolve()
    cards = []
    for path in sorted({Path(p).resolve() for p in paths if Path(p).suffix.lower() == '.png'}):
        relative = path.relative_to(output_dir).as_posix()
        name = escape(LABELS.get(path.stem, path.stem))
        links = ' · '.join(f'<a href="{escape(path.with_suffix(suffix).relative_to(output_dir).as_posix())}">{suffix[1:].upper()}</a>'
                           for suffix in ['.png', '.svg', '.pdf'] if path.with_suffix(suffix).exists())
        cards.append(f'<article><a href="{escape(relative)}"><img src="{escape(relative)}" alt="{name}" loading="lazy"></a><p>{name}</p><small>{links}</small></article>')
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>植物分布实验图片</title>'
            '<style>body{font-family:Arial,"Microsoft YaHei",sans-serif;background:#f3f6f3;color:#243b36;margin:30px}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:24px}'
            'article{background:white;padding:18px;border:1px solid #d8e4db;border-radius:10px}'
            'img{width:100%;height:280px;object-fit:contain}p{margin:12px 0 6px}a{color:#2f6b57}'
            'small{line-height:1.8}</style><h1>植物分布实验图片</h1>'
            '<p>本次运行生成的图片；点击查看原图，SVG / PDF 可用于论文排版。</p><main>' + ''.join(cards) + '</main></html>')
    destination = output_dir / 'index.html'
    destination.write_text(html, encoding='utf-8')
    return destination
