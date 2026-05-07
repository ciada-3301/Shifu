---
name: xlsx
description: "Work with Excel and CSV spreadsheets — create, read, edit, format, restructure, or convert them. Trigger when the mission involves any .xlsx, .csv, or .tsv file: building tables from scratch, updating cells or ranges, adding/removing sheets, bulk data edits, formula insertion, or exporting to CSV."
tags: excel, spreadsheet, xlsx, csv, tsv, data, tables
---

# Excel / Spreadsheet Skill

## What this skill covers
Use this skill whenever the mission involves a `.xlsx`, `.csv`, or `.tsv` file as input or output.  
You have one dedicated tool: **`excel_tool`**.  
All files live inside `Playground/`.

---

## Tool Reference — `excel_tool`

Every operation is one call to `excel_tool` with a different `action` string.

### Quick-reference table

| action | Purpose | Key extra params |
|---|---|---|
| `read` | Dump sheet rows as JSON | `sheet_name`, `preview` (default 20, 0 = all) |
| `read_csv` | Read a .csv / .tsv as JSON | `preview` |
| `write` | Create a new workbook from JSON data | `data`, `style_json` |
| `edit` | Update one or many cells | `cell`+`value` OR `data` list |
| `add_sheet` | Add a new sheet (optionally with data) | `new_name`, `data` |
| `delete_sheet` | Remove a sheet | `sheet_name` |
| `rename_sheet` | Rename a sheet | `sheet_name`, `new_name` |
| `insert_rows` | Insert blank rows | `start_row`, `num_rows` |
| `delete_rows` | Delete rows | `start_row`, `num_rows` |
| `insert_cols` | Insert blank columns | `start_col`, `num_cols` |
| `delete_cols` | Delete columns | `start_col`, `num_cols` |
| `get_info` | Sheet names, dimensions, named ranges | — |
| `to_csv` | Export a sheet to CSV | `output_path` |
| `from_csv` | Import a CSV as a new sheet in xlsx | `output_path` (target xlsx) |

---

## Step-by-step workflows

### 1 · Read an existing file

```
Step 1: excel_tool(action="get_info", path="Playground/data.xlsx")
        → confirms sheet names and dimensions before reading

Step 2: excel_tool(action="read", path="Playground/data.xlsx",
                   sheet_name="Sales", preview=50)
        → returns JSON rows; use this to understand the data
          before making edits
```

**Tips**
- Always call `get_info` first on an unfamiliar file.
- Set `preview=0` only when you genuinely need every row; large files slow the loop.
- Cross-sheet references in formulas use the format `SheetName!A1`.

---

### 2 · Create a new workbook

```
Step 1: Build a JSON list of dicts — one dict per row, keys = column headers.
        Single sheet:
          data = '[{"Month":"Jan","Revenue":12000,"Cost":8000}, ...]'

        Multiple sheets:
          data = '{"sheets": {"Revenue": [...], "Costs": [...]}}'

Step 2: excel_tool(action="write",
                   path="Playground/report.xlsx",
                   data=<json string>,
                   style_json='{"bold": true, "bg_color": "2E4057", "color": "FFFFFF"}')
        → header row gets styled automatically
```

**Tips**
- Column headers come from the dict keys of the first row.
- Values starting with `=` are written as live Excel formulas — use them.
- `style_json` keys: `bold` (bool), `italic` (bool), `color` (hex), `bg_color` (hex), `align` ("left"/"center"/"right").

---

### 3 · Edit cells in an existing file

**Single cell (shorthand)**
```
excel_tool(action="edit", path="Playground/report.xlsx",
           cell="B2", value="=SUM(B3:B15)")
```

**Many cells at once (bulk)**
```
excel_tool(action="edit", path="Playground/report.xlsx",
           data='[
             {"cell": "A1", "value": "Updated Title",
              "style": {"bold": true, "bg_color": "FF0000"}},
             {"cell": "C5", "value": "=C3/C4"},
             {"cell": "D5", "value": "99.5"}
           ]')
```

**Rules**
- Prefer bulk edits (one tool call) over many single-cell calls.
- Strings starting with `=` become formulas — never hardcode a computed number.
- Cell references are always in A1 notation (`B3`, `$B$3`, `Sheet2!B3`).

---

### 4 · Sheet management

```
# Add a summary sheet
excel_tool(action="add_sheet", path="Playground/report.xlsx", new_name="Summary")

# Populate it
excel_tool(action="edit", path="Playground/report.xlsx",
           sheet_name="Summary",
           data='[{"cell":"A1","value":"Total"},{"cell":"B1","value":"=Sales!B20"}]')

# Rename a sheet
excel_tool(action="rename_sheet", path="Playground/report.xlsx",
           sheet_name="Sheet1", new_name="Raw Data")

# Delete a sheet
excel_tool(action="delete_sheet", path="Playground/report.xlsx",
           sheet_name="Temp")
```

---

### 5 · Row and column operations

```
# Insert 3 blank rows before row 5
excel_tool(action="insert_rows", path="Playground/report.xlsx",
           sheet_name="Sales", start_row=5, num_rows=3)

# Delete rows 10–12
excel_tool(action="delete_rows", path="Playground/report.xlsx",
           sheet_name="Sales", start_row=10, num_rows=3)

# Insert 2 blank columns before column 3 (= column C)
excel_tool(action="insert_cols", path="Playground/report.xlsx",
           start_col=3, num_cols=2)
```

---

### 6 · CSV workflows

```
# Read a CSV into JSON (inspect before converting)
excel_tool(action="read_csv", path="Playground/export.csv", preview=30)

# Import a CSV as a new sheet inside an xlsx
excel_tool(action="from_csv",
           path="Playground/export.csv",
           new_name="Imported",
           output_path="Playground/master.xlsx")

# Export one sheet to CSV
excel_tool(action="to_csv",
           path="Playground/master.xlsx",
           sheet_name="Sales",
           output_path="Playground/sales_export.csv")
```

---

## Formula guidelines

| Rule | Example |
|---|---|
| Use Excel formulas, not hardcoded numbers | `=SUM(B2:B9)` not `4500` |
| Reference cells, not literal values | `=B5*(1+$B$6)` not `=B5*1.05` |
| Guard against divide-by-zero | `=IF(C4=0,"–",B4/C4)` |
| Cross-sheet references | `=Revenue!B10` |
| Percentage display | value is `0.15`; format cell as `0.0%` |

---

## File & path conventions

- All files go inside `Playground/`  — use relative paths like `"Playground/report.xlsx"`.
- The tool resolves relative paths to `Playground/` automatically when the file is not found at the literal path.
- Never write files outside `Playground/` unless the user explicitly requests it.

---

## Common mistake checklist

- **Don't** call `read` before `get_info` on an unknown file — you may read the wrong sheet.
- **Don't** use `data_only=True` semantics for edits — the tool handles this correctly internally; just pass formulas as strings.
- **Don't** make ten single-cell edit calls when one bulk `data` list will do.
- **Don't** hardcode a Python-computed number into a cell — write an Excel formula instead.
- **Do** end the mission with `✅ DONE:` and a one-line summary of what was created or changed.

---

## Mission summary template

End every excel mission like this:

```
✅ DONE: Created Playground/report.xlsx with 3 sheets (Revenue, Costs, Summary).
         Revenue sheet: 24 rows of monthly data, formulas in column D for margin %.
         Summary sheet: cross-sheet totals pulling from Revenue!B25 and Costs!B25.
```