use tree_sitter::{Language, Node, Parser};

/// Extract code symbols using tree-sitter AST parsing.
/// Returns structured outline entries (function signatures, struct/class names,
/// trait/interface definitions, impl blocks, etc.)
/// Falls back to None if the language isn't supported.
pub fn extract_outline_tree_sitter(content: &str, extension: &str) -> Option<Vec<String>> {
    let language = language_for_extension(extension)?;
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .expect("language version mismatch");
    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    let entries = match extension {
        "rs" => extract_rust(root, content),
        "py" => extract_python(root, content),
        "js" | "jsx" => extract_javascript(root, content),
        "ts" | "tsx" => extract_typescript(root, content),
        "go" => extract_go(root, content),
        _ => return None,
    };

    Some(entries)
}

fn language_for_extension(ext: &str) -> Option<Language> {
    match ext {
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        _ => None,
    }
}

fn node_text<'a>(node: Node<'a>, src: &'a str) -> &'a str {
    node.utf8_text(src.as_bytes()).unwrap_or("")
}

/// Grab text from `node` up to (but not including) the first child matching `stop_kind`,
/// or just the node's own text if no such child exists. Trims trailing whitespace.
fn text_until_child(node: Node<'_>, src: &str, stop_kind: &str) -> String {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) 
            && child.kind() == stop_kind {
            let start = node.start_byte();
            let end = child.start_byte();
            return src[start..end].trim_end().to_string();
        }
    }
    first_line(node_text(node, src))
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// Rust
// ---------------------------------------------------------------------------

fn extract_rust(root: Node<'_>, src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        extract_rust_node(child, src, &mut out, 0);
    }
    out.truncate(60);
    out
}

fn extract_rust_node(node: Node<'_>, src: &str, out: &mut Vec<String>, depth: usize) {
    if depth > 1 {
        return;
    }

    let indent = if depth > 0 { "  " } else { "" };

    match node.kind() {
        "function_item" => {
            let sig = text_until_child(node, src, "block");
            out.push(format!("{indent}{sig}"));
        }
        "struct_item" => {
            let sig = text_until_child(node, src, "field_declaration_list");
            // Fallback for unit structs / tuple structs without braces
            let sig = if sig == node_text(node, src).trim_end() {
                first_line(node_text(node, src))
            } else {
                sig
            };
            out.push(format!("{indent}{sig}"));
        }
        "enum_item" => {
            let sig = text_until_child(node, src, "enum_variant_list");
            out.push(format!("{indent}{sig}"));
        }
        "trait_item" => {
            let sig = text_until_child(node, src, "declaration_list");
            out.push(format!("{indent}{sig}"));
        }
        "impl_item" => {
            let sig = text_until_child(node, src, "declaration_list");
            out.push(format!("{indent}{sig}"));
            // Extract methods one level deep
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "declaration_list" {
                    let mut inner_cursor = child.walk();
                    for item in child.children(&mut inner_cursor) {
                        extract_rust_node(item, src, out, depth + 1);
                    }
                }
            }
        }
        "mod_item" => {
            let name = named_child_text(node, "identifier", src);
            if let Some(name) = name {
                let vis = visibility_prefix(node, src);
                out.push(format!("{indent}{vis}mod {name}"));
            }
        }
        "type_item" => {
            let sig = first_line(node_text(node, src));
            out.push(format!("{indent}{sig}"));
        }
        "const_item" | "static_item" => {
            let sig = first_line(node_text(node, src));
            out.push(format!("{indent}{sig}"));
        }
        // Attribute items wrap the real item (e.g. #[derive(...)] struct Foo)
        "attribute_item" => {}
        _ => {}
    }
}

fn visibility_prefix<'a>(node: Node<'a>, src: &'a str) -> &'a str {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            let text = node_text(child, src);
            if text == "pub" {
                return "pub ";
            }
            // pub(crate), pub(super), etc.
            return "";
        }
    }
    ""
}

fn named_child_text<'a>(node: Node<'a>, field: &str, src: &'a str) -> Option<&'a str> {
    node.child_by_field_name(field).map(|n| node_text(n, src))
}

// ---------------------------------------------------------------------------
// Python
// ---------------------------------------------------------------------------

fn extract_python(root: Node<'_>, src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        extract_python_node(child, src, &mut out, 0);
    }
    out.truncate(60);
    out
}

fn extract_python_node(node: Node<'_>, src: &str, out: &mut Vec<String>, depth: usize) {
    if depth > 1 {
        return;
    }

    let indent = if depth > 0 { "  " } else { "" };

    match node.kind() {
        "function_definition" => {
            let sig = python_function_sig(node, src);
            out.push(format!("{indent}{sig}"));
        }
        "class_definition" => {
            let name = named_child_text(node, "name", src).unwrap_or("?");
            let superclasses = node
                .child_by_field_name("superclasses")
                .map(|n| node_text(n, src))
                .unwrap_or("");
            out.push(format!("{indent}class {name}{superclasses}"));
            // Methods
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "block" {
                    let mut inner = child.walk();
                    for item in child.children(&mut inner) {
                        extract_python_node(item, src, out, depth + 1);
                    }
                }
            }
        }
        "decorated_definition" => {
            // Walk through to find the actual definition inside
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "function_definition" || child.kind() == "class_definition" {
                    extract_python_node(child, src, out, depth);
                }
            }
        }
        _ => {}
    }
}

fn python_function_sig(node: Node<'_>, src: &str) -> String {
    let name = named_child_text(node, "name", src).unwrap_or("?");
    let params = node
        .child_by_field_name("parameters")
        .map(|n| node_text(n, src))
        .unwrap_or("()");

    // Check if async by looking at the source text prefix
    let node_start = node.start_byte();
    let prefix = &src[..node_start];
    let is_async =
        prefix.trim_end().ends_with("async") || node_text(node, src).starts_with("async");

    if is_async {
        format!("async def {name}{params}")
    } else {
        format!("def {name}{params}")
    }
}

// ---------------------------------------------------------------------------
// JavaScript
// ---------------------------------------------------------------------------

fn extract_javascript(root: Node<'_>, src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        extract_js_node(child, src, &mut out, 0, false);
    }
    out.truncate(60);
    out
}

fn extract_js_node(node: Node<'_>, src: &str, out: &mut Vec<String>, depth: usize, is_ts: bool) {
    if depth > 1 {
        return;
    }

    let indent = if depth > 0 { "  " } else { "" };

    match node.kind() {
        "function_declaration" => {
            let name = named_child_text(node, "name", src).unwrap_or("?");
            let params = node
                .child_by_field_name("parameters")
                .map(|n| node_text(n, src))
                .unwrap_or("()");
            out.push(format!("{indent}function {name}{params}"));
        }
        "class_declaration" => {
            let name = named_child_text(node, "name", src).unwrap_or("?");
            out.push(format!("{indent}class {name}"));
            // Methods
            if let Some(body) = node.child_by_field_name("body") {
                let mut cursor = body.walk();
                for child in body.children(&mut cursor) {
                    if child.kind() == "method_definition" {
                        let method_name = named_child_text(child, "name", src).unwrap_or("?");
                        out.push(format!("  {method_name}()"));
                    }
                }
            }
        }
        "lexical_declaration" => {
            // const foo = (...) => { ... }
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "variable_declarator" 
                    && let Some(value) = child.child_by_field_name("value")
                    && value.kind() == "arrow_function" {
                    let name = named_child_text(child, "name", src).unwrap_or("?");
                    let params = value
                        .child_by_field_name("parameters")
                        .map(|n| node_text(n, src));
                    let kw = keyword_for_lexical(node, src);
                    if let Some(params) = params {
                        out.push(format!("{indent}{kw} {name} = {params} =>"));
                    } else {
                        out.push(format!("{indent}{kw} {name} = () =>"));
                    }
                }
            }
        }
        "export_statement" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() != "export" {
                    extract_js_node(child, src, out, depth, is_ts);
                }
            }
        }
        // TypeScript-specific nodes
        "interface_declaration" if is_ts => {
            let name = named_child_text(node, "name", src).unwrap_or("?");
            out.push(format!("{indent}interface {name}"));
        }
        "type_alias_declaration" if is_ts => {
            let name = named_child_text(node, "name", src).unwrap_or("?");
            out.push(format!("{indent}type {name}"));
        }
        "enum_declaration" if is_ts => {
            let name = named_child_text(node, "name", src).unwrap_or("?");
            out.push(format!("{indent}enum {name}"));
        }
        _ => {}
    }
}

fn keyword_for_lexical<'a>(node: Node<'a>, src: &'a str) -> &'a str {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if kind == "const" || kind == "let" || kind == "var" {
            return node_text(child, src);
        }
    }
    "const"
}

// ---------------------------------------------------------------------------
// TypeScript (reuses JS extraction with TS flag)
// ---------------------------------------------------------------------------

fn extract_typescript(root: Node<'_>, src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        extract_js_node(child, src, &mut out, 0, true);
    }
    out.truncate(60);
    out
}

// ---------------------------------------------------------------------------
// Go
// ---------------------------------------------------------------------------

fn extract_go(root: Node<'_>, src: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        extract_go_node(child, src, &mut out);
    }
    out.truncate(60);
    out
}

fn extract_go_node(node: Node<'_>, src: &str, out: &mut Vec<String>) {
    match node.kind() {
        "function_declaration" => {
            let sig = text_until_child(node, src, "block");
            out.push(sig);
        }
        "method_declaration" => {
            let sig = text_until_child(node, src, "block");
            out.push(sig);
        }
        "type_declaration" => {
            // type_declaration contains type_spec children
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if child.kind() == "type_spec" {
                    let name = named_child_text(child, "name", src).unwrap_or("?");
                    let type_node = child.child_by_field_name("type");
                    let kind = type_node.map(|n| n.kind()).unwrap_or("");
                    match kind {
                        "struct_type" => out.push(format!("type {name} struct")),
                        "interface_type" => out.push(format!("type {name} interface")),
                        _ => {
                            let type_text = type_node
                                .map(|n| first_line(node_text(n, src)))
                                .unwrap_or_default();
                            out.push(format!("type {name} {type_text}"));
                        }
                    }
                }
            }
        }
        _ => {}
    }
}
