"""
Configuration du solveur CBC pour l'app PyInstaller
"""
import os
import sys
import stat
import pulp

def setup_solver():
    """
    Configure le chemin du solveur CBC, en particulier pour les applications
    compilées avec PyInstaller.
    """
    solver_path = None
    
    # Détecter si l'application est compilée (frozen)
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Chemin dans une app macOS compilée
        # Le solveur est copié dans Contents/Resources/solverbin/
        base_path = sys._MEIPASS
        
        # Tenter plusieurs chemins possibles à l'intérieur du bundle
        possible_paths = [
            os.path.join(base_path, 'solverbin', 'cbc'),
            os.path.join(base_path, 'cbc'),
            # Chemin si les ressources sont dans un sous-dossier
            os.path.join(base_path, 'Resources', 'solverbin', 'cbc')
        ]
        
        print(f"INFO: Application compilée détectée. Base de recherche: {base_path}")
        
        for path in possible_paths:
            if os.path.exists(path):
                solver_path = path
                print(f"INFO: Solveur trouvé à: {solver_path}")
                break
        
        if solver_path:
            # Rendre le fichier exécutable (important sur macOS/Linux)
            try:
                st = os.stat(solver_path)
                os.chmod(solver_path, st.st_mode | stat.S_IEXEC)
                print("INFO: Permissions d'exécution vérifiées.")
            except Exception as e:
                print(f"AVERTISSEMENT: Impossible de changer les permissions: {e}")
        else:
            print("ERREUR: Le solveur CBC n'a pas été trouvé dans le bundle de l'application.")
            
    else:
        # En mode développement, chercher localement
        local_path = os.path.join(os.path.dirname(__file__), 'solverbin', 'cbc')
        if os.path.exists(local_path):
            solver_path = local_path
            print(f"INFO: Mode développement. Solveur trouvé à: {solver_path}")

    # Si un chemin a été trouvé, le passer à PuLP
    if solver_path and os.path.exists(solver_path):
        # Créer une instance du solveur avec le chemin spécifié
        try:
            solver = pulp.COIN_CMD(path=solver_path, msg=0)
            
            # Remplacer le solveur par défaut de PuLP
            original_get_solver = pulp.getSolver
            
            def custom_get_solver(name='PULP_CBC_CMD', *args, **kwargs):
                if name in ['COIN_CMD', 'PULP_CBC_CMD', 'CBC']:
                    print("INFO: Utilisation du solveur CBC configuré sur mesure.")
                    return solver
                return original_get_solver(name, *args, **kwargs)
                
            pulp.getSolver = custom_get_solver
            
            print("INFO: Le solveur CBC a été configuré avec succès pour PuLP.")
        except Exception as e:
            print(f"ERREUR lors de la configuration du solveur: {e}")
    else:
        print("AVERTISSEMENT: Aucun solveur CBC local n'a été trouvé ou configuré.")

# Exécuter la configuration au moment de l'importation de ce module
setup_solver()
