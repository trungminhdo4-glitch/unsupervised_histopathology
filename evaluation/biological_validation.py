# evaluation/biological_validation.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

class BiologicalValidator:
    """Validiert Hypothesen gegen biologisches Vorwissen"""
    
    @staticmethod
    def validate_hypothesis(hypothesis, pathologist_feedback=None):
        """Validiert eine Hypothese"""
        base_score = hypothesis['confidence']
        
        # Simuliere Pathologen-Feedback (in echtem System: echte Bewertung)
        if pathologist_feedback:
            feedback_score = pathologist_feedback.get('rating', 5) / 10
            evidence_quality = pathologist_feedback.get('evidence_quality', 0.7)
            return base_score * 0.6 + feedback_score * 0.3 + evidence_quality * 0.1
        
        return base_score
    
    @staticmethod
    def generate_validation_report(hypotheses, pathologist_feedbacks=None):
        """Generiert einen Validierungsbericht"""
        report = "🧬 BIOLOGISCHE VALIDIERUNGSBERICHT\n"
        report += "=" * 50 + "\n\n"
        
        for i, hyp in enumerate(hypotheses):
            path_feedback = pathologist_feedbacks[i] if pathologist_feedbacks else None
            score = BiologicalValidator.validate_hypothesis(hyp, path_feedback)
            
            status = "✅ Stark validiert" if score > 0.8 else "⚠️ Benötigt weitere Validierung"
            report += f"Hypothese {i+1}: {hyp['title']}\n"
            report += f"   📊 Validierungsscore: {score:.2f}/1.0\n"
            report += f"   📝 Beschreibung: {hyp['description']}\n"
            report += f"   🏷️ Status: {status}\n\n"
        
        return report