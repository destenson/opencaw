use caw_core::{CawResult, ProbeMarker, Range, ThinkingStep};
use regex::Regex;

/// Extract probe markers from model output
pub fn extract_probes(text: &str) -> Vec<ProbeMarker> {
    let pattern = Regex::new(r"<probe>(.*?)</probe>").unwrap();
    pattern
        .captures_iter(text)
        .enumerate()
        .map(|(idx, cap)| ProbeMarker {
            content: cap.get(1).unwrap().as_str().to_string(),
            position: cap.get(0).unwrap().start(),
        })
        .collect()
}

/// Extract thinking steps from reasoning model output
pub fn extract_thinking_steps(text: &str) -> Vec<ThinkingStep> {
    // Look for <think>...</think> tags
    let think_pattern = Regex::new(r"<think>(.*?)</think>").unwrap();
    let mut steps = Vec::new();
    
    for (idx, cap) in think_pattern.captures_iter(text).enumerate() {
        steps.push(ThinkingStep {
            content: cap.get(1).unwrap().as_str().to_string(),
            step_number: idx,
        });
    }
    
    // If no explicit tags, try to detect step boundaries heuristically
    if steps.is_empty() {
        steps = detect_step_boundaries(text);
    }
    
    steps
}

/// Heuristic detection of thinking step boundaries
fn detect_step_boundaries(text: &str) -> Vec<ThinkingStep> {
    let mut steps = Vec::new();
    let mut current_step = String::new();
    let mut step_number = 0;
    
    for line in text.lines() {
        // Step boundary heuristics:
        // - Double newline
        // - Numbered list item
        // - "Therefore", "Thus", "So" at start
        let is_boundary = line.trim().is_empty() 
            || line.trim_start().starts_with(|c: char| c.is_numeric() && line.contains('.'))
            || line.trim_start().starts_with("Therefore")
            || line.trim_start().starts_with("Thus")
            || line.trim_start().starts_with("So");
        
        if is_boundary && !current_step.is_empty() {
            steps.push(ThinkingStep {
                content: current_step.trim().to_string(),
                step_number,
            });
            step_number += 1;
            current_step.clear();
        } else if !line.trim().is_empty() {
            current_step.push_str(line);
            current_step.push('\n');
        }
    }
    
    // Add final step
    if !current_step.is_empty() {
        steps.push(ThinkingStep {
            content: current_step.trim().to_string(),
            step_number,
        });
    }
    
    // Minimum step length to avoid noise
    steps.retain(|s| s.content.len() > 20);
    
    steps
}

/// Apply range selection to content
pub fn apply_range(content: &str, range: &Range) -> CawResult<String> {
    match range {
        Range::Full => Ok(content.to_string()),
        
        Range::Lines { start, end } => {
            let lines: Vec<&str> = content.lines().collect();
            let start_idx = start.saturating_sub(1); // 1-indexed to 0-indexed
            let end_idx = (*end).min(lines.len());
            
            if start_idx >= lines.len() {
                return Ok(String::new());
            }
            
            Ok(lines[start_idx..end_idx].join("\n"))
        }
        
        Range::Heading { path } => {
            // Simple heading extraction for markdown
            extract_heading_section(content, path)
        }
        
        Range::Tokens { start, count } => {
            // Approximate token selection (char/4 heuristic)
            let char_start = start * 4;
            let char_count = count * 4;
            let chars: Vec<char> = content.chars().collect();
            
            if char_start >= chars.len() {
                return Ok(String::new());
            }
            
            let end = (char_start + char_count).min(chars.len());
            Ok(chars[char_start..end].iter().collect())
        }
        
        Range::Custom(spec) => {
            // For now, treat custom ranges as full content
            // Could extend with regex patterns, etc.
            Ok(content.to_string())
        }
    }
}

/// Extract content under a heading path (simplified for MVP)
fn extract_heading_section(content: &str, path: &[String]) -> CawResult<String> {
    if path.is_empty() {
        return Ok(content.to_string());
    }
    
    let heading_pattern = format!(r"(?m)^#{{{level}}}\s+{name}\s*$", 
        level = path.len(), 
        name = regex::escape(&path[path.len() - 1])
    );
    
    let re = Regex::new(&heading_pattern).map_err(|e| {
        caw_core::CawError::InvalidInput(format!("Invalid heading pattern: {}", e))
    })?;
    
    if let Some(mat) = re.find(content) {
        let start = mat.end();
        
        // Find next heading at same or higher level
        let next_heading = format!(r"(?m)^#{{{},{}}}[^#]", 1, path.len());
        let next_re = Regex::new(&next_heading).unwrap();
        
        let end = next_re
            .find(&content[start..])
            .map(|m| start + m.start())
            .unwrap_or(content.len());
        
        Ok(content[start..end].trim().to_string())
    } else {
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_probes() {
        let text = "Consider <probe>check config.toml</probe> and <probe>review main.rs</probe>";
        let probes = extract_probes(text);
        assert_eq!(probes.len(), 2);
        assert_eq!(probes[0].content, "check config.toml");
        assert_eq!(probes[1].content, "review main.rs");
    }
    
    #[test]
    fn test_range_lines() {
        let content = "line1\nline2\nline3\nline4\nline5";
        let range = Range::Lines { start: 2, end: 4 };
        let result = apply_range(content, &range).unwrap();
        assert_eq!(result, "line2\nline3\nline4");
    }
    
    #[test]
    fn test_range_parse() {
        assert!(matches!(Range::parse("full"), Range::Full));
        assert!(matches!(Range::parse("10-20"), Range::Lines { start: 10, end: 20 }));
        assert!(matches!(Range::parse("L5-L15"), Range::Lines { start: 5, end: 15 }));
        assert!(matches!(Range::parse("#Section/Subsection"), Range::Heading { .. }));
        assert!(matches!(Range::parse("T100:500"), Range::Tokens { start: 100, count: 500 }));
    }
}
