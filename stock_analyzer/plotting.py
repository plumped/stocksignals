import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
import io
import base64

def plot_signal_influences_with_rating(result):
    signals = result['signals']
    recommendation = result['recommendation']
    score = result['score']

    labels = []
    values = []
    colors = []

    for signal in signals:
        label, action, description, influence = signal
        labels.append(label)
        try:
            influence = float(influence)
        except (ValueError, TypeError):
            influence = 0

        values.append(influence)
        colors.append('green' if influence >= 0 else 'red')

    recommendation_colors = {
        "STRONG BUY": "darkgreen",
        "BUY": "green",
        "HOLD": "orange",
        "SELL": "red",
        "STRONG SELL": "darkred"
    }
    rec_color = recommendation_colors.get(recommendation, "black")

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(labels, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom')

    ax.set_ylabel('Einfluss auf Score')
    ax.set_title('Signalbeiträge zum technischen Score', fontsize=16)
    ax.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45, ha='right')

    plt.text(
        0.5, 1.15,
        f"Gesamtbewertung: {recommendation} (Score: {score:.2f})",
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        color=rec_color
    )

    plt.tight_layout()

    # 🧠 Statt plt.show(): Bild speichern
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)

    # Base64-kodieren
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic
