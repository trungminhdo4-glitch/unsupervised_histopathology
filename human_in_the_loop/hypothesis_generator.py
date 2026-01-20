# human_in_the_loop/hypothesis_generator.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

class HypothesisGenerator:
    """Generiert biologische Hypothesen aus latenten Repräsentationen"""
    
    @staticmethod
    def generate_hypotheses(latent_dim, feature_name, correlation):
        """Generiert eine Hypothese basierend auf Korrelation"""
        hypotheses = {
            'nuclear_size': (
                "Zellkern-Größe",
                f"Hohe Werte in Dimension {latent_dim} korrelieren mit größeren Zellkernen. "
                "Dies könnte auf einen aggressiveren Tumor-Subtyp hindeuten, "
                "da vergrößerte Kerne oft mit hoher Teilungsaktivität assoziiert sind."
            ),
            'nuclear_density': (
                "Zelldichte",
                f"Hohe Werte in Dimension {latent_dim} zeigen eine höhere Zelldichte. "
                "Dies könnte auf invasives Wachstum hindeuten, ein Merkmal "
                "fortgeschrittener Karzinome."
            ),
            'tissue_structure': (
                "Gewebearchitektur",
                f"Dimension {latent_dim} trennt strukturierte von unstrukturierten Geweben. "
                "Unstrukturierte Muster könnten auf einen Verlust der normalen "
                "Gewebeorganisation hindeuten – ein Zeichen für Malignität."
            )
        }
        
        if feature_name in hypotheses:
            title, description = hypotheses[feature_name]
            return {
                'title': title,
                'description': description,
                'confidence': min(0.95, abs(correlation) * 1.2)  # Skaliert auf [0, 0.95]
            }
        return None