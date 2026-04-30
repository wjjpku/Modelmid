import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_matplotlib_fonts():
    """Use an installed CJK-capable font when available."""
    plt.style.use('seaborn-v0_8-whitegrid')

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    candidates = [
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'Arial Unicode MS',
    ]
    selected_fonts = [name for name in candidates if name in available_fonts]
    if not selected_fonts:
        selected_fonts = ['DejaVu Sans']

    plt.rcParams['font.sans-serif'] = selected_fonts + ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False

    return selected_fonts
