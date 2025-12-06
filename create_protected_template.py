"""
Script pour créer un template Excel protégé avec validation des données.
- Validation stricte : colonnes P1-P10 acceptent uniquement 0 ou 1
- Protection : seul le Tableau 1 (disponibilités) est modifiable
"""

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side, Protection
from openpyxl.worksheet.datavalidation import DataValidation

def create_protected_template(filename="template_greve.xlsx", num_teachers=50):
    """
    Créer un template Excel protégé avec validation des données
    
    Args:
        filename: Nom du fichier à créer
        num_teachers: Nombre d'enseignants dans le template
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Données Grèves"
    
    # Styles
    header_fill = PatternFill(start_color="00D9FF", end_color="00D9FF", fill_type="solid")
    header_font = Font(bold=True, size=12, color="0A0E27")
    center_align = Alignment(horizontal="center", vertical="center")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Titre principal
    ws.merge_cells('A1:L1')
    ws['A1'] = "OPTIMISATEUR DE GRÈVE - TABLEAU 1 : DISPONIBILITÉS"
    ws['A1'].font = Font(bold=True, size=14, color="00D9FF")
    ws['A1'].alignment = center_align
    
    # Instructions
    ws.merge_cells('A2:L2')
    ws['A2'] = "Remplissez uniquement les colonnes P1 à P10 avec 0 (pas disponible) ou 1 (disponible)"
    ws['A2'].font = Font(italic=True, size=10, color="FF0000")
    ws['A2'].alignment = center_align
    
    # En-têtes
    headers = ['Prénom', 'Nom'] + [f'P{i}' for i in range(1, 11)]
    for col_idx, header in enumerate(headers, start=1):
        cell = ws.cell(row=3, column=col_idx)
        cell.value = header
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
        cell.border = border
    
    # Données exemples (lignes 4 à 4+num_teachers-1)
    for row_idx in range(4, 4 + num_teachers):
        # Prénom et Nom
        ws.cell(row=row_idx, column=1).value = f"Prénom{row_idx-3}"
        ws.cell(row=row_idx, column=2).value = f"Nom{row_idx-3}"
        
        # P1 à P10 : valeurs par défaut 0
        for col_idx in range(3, 13):
            cell = ws.cell(row=row_idx, column=col_idx)
            cell.value = 0
            cell.alignment = center_align
            cell.border = border
    
    # VALIDATION DES DONNÉES : P1 à P10 doivent être 0 ou 1
    # Créer une validation de liste pour 0 ou 1
    dv = DataValidation(
        type="list",
        formula1='"0,1"',
        allow_blank=False,
        showErrorMessage=True,
        errorTitle="Valeur invalide",
        error="Seules les valeurs 0 ou 1 sont autorisées.\n0 = pas disponible\n1 = disponible"
    )
    
    # Appliquer la validation aux colonnes P1 à P10 (colonnes C à L)
    # Pour toutes les lignes de données (ligne 4 à 4+num_teachers-1)
    dv.add(f'C4:L{3 + num_teachers}')
    ws.add_data_validation(dv)
    
    # PROTECTION DE LA FEUILLE
    # 1. Verrouiller toutes les cellules par défaut
    for row in ws.iter_rows():
        for cell in row:
            cell.protection = Protection(locked=True)
    
    # 2. Déverrouiller UNIQUEMENT les cellules du Tableau 1 (P1 à P10)
    for row_idx in range(4, 4 + num_teachers):
        # Déverrouiller Prénom et Nom
        ws.cell(row=row_idx, column=1).protection = Protection(locked=False)
        ws.cell(row=row_idx, column=2).protection = Protection(locked=False)
        
        # Déverrouiller P1 à P10
        for col_idx in range(3, 13):
            ws.cell(row=row_idx, column=col_idx).protection = Protection(locked=False)
    
    # 3. Activer la protection de la feuille (sans mot de passe pour faciliter)
    ws.protection.sheet = True
    ws.protection.enable()
    
    # Ajuster les largeurs de colonnes
    ws.column_dimensions['A'].width = 15
    ws.column_dimensions['B'].width = 15
    for col in ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
        ws.column_dimensions[col].width = 8
    
    # Note d'information en bas
    info_row = 4 + num_teachers + 2
    ws.merge_cells(f'A{info_row}:L{info_row}')
    ws[f'A{info_row}'] = "⚠️ PROTECTION ACTIVÉE : Seules les cellules du Tableau 1 peuvent être modifiées (Prénom, Nom, P1-P10)"
    ws[f'A{info_row}'].font = Font(italic=True, size=9, color="666666")
    ws[f'A{info_row}'].alignment = center_align
    
    # Sauvegarder
    wb.save(filename)
    print(f"✅ Template protégé créé : {filename}")
    print(f"   - {num_teachers} enseignants")
    print(f"   - Validation des données : P1-P10 acceptent uniquement 0 ou 1")
    print(f"   - Protection : seul le Tableau 1 est modifiable")


if __name__ == "__main__":
    # Créer le template vide (50 enseignants)
    create_protected_template("template_greve.xlsx", num_teachers=50)
    
    # Créer le template de test (50 enseignants avec données variées)
    create_protected_template("template_greve_test_50.xlsx", num_teachers=50)
    
    print("\n🎯 Templates créés avec succès !")
