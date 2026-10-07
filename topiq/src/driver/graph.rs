//! Which units depend on which.
//!
//! A unit is analysed against the interfaces of the units it depends on, so
//! those are analysed first. Units that depend on one another, directly or
//! through others, cannot each come first: they form one group, analysed
//! together. [`Graph::groups`] gives the groups in an order where each comes
//! after every group it depends on.
//!
//! [`Graph::render`] draws the graph as text, for `tqc emit --stage
//! depgraph`: each unit given, with the units it depends on below it, then
//! the order the groups are analysed in.

use std::fmt::Write;

/// Units and the units each depends on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Graph {
    nodes: Vec<Node>,
}

/// One unit of a graph.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Node {
    name: String,
    // what the drawing says of the unit beside its name, such as `library`
    notes: Vec<String>,
    deps: Vec<usize>,
}

impl Graph {
    /// A graph of no units.
    pub fn new() -> Graph {
        Graph::default()
    }

    /// How many units the graph holds.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph holds no unit.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The unit called `name`.
    pub fn add(&mut self, name: &str) -> usize {
        match self.index(name) {
            Some(i) => i,
            None => self.push(name),
        }
    }

    /// A new unit called `name`, which is the graph's last, even if another
    /// has that name: so that a graph's units can be numbered as the items
    /// of a list they are made from.
    pub fn push(&mut self, name: &str) -> usize {
        self.nodes.push(Node {
            name: name.to_owned(),
            notes: Vec::new(),
            deps: Vec::new(),
        });
        self.nodes.len() - 1
    }

    /// The unit called `name`.
    pub fn index(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.name == name)
    }

    /// The name of unit `i`.
    pub fn name(&self, i: usize) -> &str {
        &self.nodes[i].name
    }

    /// Records that unit `from` depends on unit `to`, once.
    pub fn link(&mut self, from: usize, to: usize) {
        if !self.nodes[from].deps.contains(&to) {
            self.nodes[from].deps.push(to);
        }
    }

    /// The units unit `i` depends on.
    pub fn deps(&self, i: usize) -> &[usize] {
        &self.nodes[i].deps
    }

    /// Adds `note` to what the drawing says of unit `i`.
    pub fn note(&mut self, i: usize, note: &str) {
        self.nodes[i].notes.push(note.to_owned());
    }

    /// The units in groups, each group after every group it depends on. A
    /// group is one unit, or several that depend on one another; its units
    /// are in the order they were added.
    pub fn groups(&self) -> Vec<Vec<usize>> {
        // Tarjan's algorithm: a group is finished only after every group it
        // reaches, so groups come out dependencies first
        struct State<'g> {
            graph: &'g Graph,
            index: Vec<Option<usize>>,
            low: Vec<usize>,
            on_stack: Vec<bool>,
            stack: Vec<usize>,
            next: usize,
            groups: Vec<Vec<usize>>,
        }
        fn visit(s: &mut State<'_>, i: usize) {
            s.index[i] = Some(s.next);
            s.low[i] = s.next;
            s.next += 1;
            s.stack.push(i);
            s.on_stack[i] = true;
            for &j in s.graph.deps(i) {
                match s.index[j] {
                    None => {
                        visit(s, j);
                        s.low[i] = s.low[i].min(s.low[j]);
                    }
                    Some(n) if s.on_stack[j] => s.low[i] = s.low[i].min(n),
                    Some(_) => {}
                }
            }
            if Some(s.low[i]) == s.index[i] {
                let mut group = Vec::new();
                while let Some(j) = s.stack.pop() {
                    s.on_stack[j] = false;
                    group.push(j);
                    if j == i {
                        break;
                    }
                }
                group.sort_unstable();
                s.groups.push(group);
            }
        }
        let n = self.nodes.len();
        let mut s = State {
            graph: self,
            index: vec![None; n],
            low: vec![0; n],
            on_stack: vec![false; n],
            stack: Vec::new(),
            next: 0,
            groups: Vec::new(),
        };
        for i in 0..n {
            if s.index[i].is_none() {
                visit(&mut s, i);
            }
        }
        s.groups
    }

    /// Whether the units of `group` depend on one another: there are
    /// several, or the one depends on itself.
    pub fn is_cycle(&self, group: &[usize]) -> bool {
        match group {
            [i] => self.deps(*i).contains(i),
            _ => group.len() > 1,
        }
    }

    /// The graph drawn as text: each of `roots`, then each unit no root
    /// reaches, with the units it depends on indented below it. A unit drawn
    /// already, with dependencies of its own, is not drawn again but marked
    /// `(see above)`; one that is being drawn around it, `(cycle)`. The order
    /// the groups are analysed in follows, a group of units that depend on
    /// one another in braces.
    pub fn render(&self, roots: &[usize]) -> String {
        self.render_with(roots, &|i| self.name(i).to_owned())
    }

    /// [`Graph::render`], with each unit written as `label` gives it, such
    /// as a link to its page.
    pub fn render_with(&self, roots: &[usize], label: &dyn Fn(usize) -> String) -> String {
        let mut out = String::new();
        let mut drawn = vec![false; self.nodes.len()];
        let mut path = Vec::new();
        let rest = (0..self.nodes.len()).filter(|i| !roots.contains(i));
        for root in roots.iter().copied().chain(rest.collect::<Vec<_>>()) {
            if drawn[root] {
                continue;
            }
            self.draw(root, "", "", label, &mut drawn, &mut path, &mut out);
        }
        let order: Vec<String> = self
            .groups()
            .iter()
            .map(|g| {
                let names: Vec<String> = g.iter().map(|&i| label(i)).collect();
                if self.is_cycle(g) { format!("{{{}}}", names.join(", ")) } else { names.join(", ") }
            })
            .collect();
        if !order.is_empty() {
            let _ = writeln!(out, "\nanalysed in order: {}", order.join(", "));
        }
        out
    }

    /// Draws unit `i` after `lead`, and what it depends on below it, each
    /// line after `indent`.
    #[allow(clippy::too_many_arguments)]
    fn draw(
        &self,
        i: usize,
        lead: &str,
        indent: &str,
        label: &dyn Fn(usize) -> String,
        drawn: &mut [bool],
        path: &mut Vec<usize>,
        out: &mut String,
    ) {
        let node = &self.nodes[i];
        let mut line = format!("{lead}{}", label(i));
        if !node.notes.is_empty() {
            let _ = write!(line, " [{}]", node.notes.join(", "));
        }
        if path.contains(&i) {
            let _ = writeln!(out, "{line} (cycle)");
            return;
        }
        if drawn[i] && !node.deps.is_empty() {
            let _ = writeln!(out, "{line} (see above)");
            return;
        }
        let _ = writeln!(out, "{line}");
        drawn[i] = true;
        path.push(i);
        for (k, &j) in node.deps.iter().enumerate() {
            let last = k + 1 == node.deps.len();
            let (branch, more) = if last { ("`-- ", "    ") } else { ("+-- ", "|   ") };
            self.draw(j, &format!("{indent}{branch}"), &format!("{indent}{more}"), label, drawn, path, out);
        }
        path.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graph of the units `names`, unit `i` depending on each unit of
    /// `deps[i]`.
    fn graph(names: &[&str], deps: &[&[usize]]) -> Graph {
        let mut g = Graph::new();
        for n in names {
            g.add(n);
        }
        for (i, ds) in deps.iter().enumerate() {
            for &d in *ds {
                g.link(i, d);
            }
        }
        g
    }

    #[test]
    fn a_unit_comes_after_what_it_depends_on() {
        let g = graph(&["app", "geo", "core"], &[&[1, 2], &[2], &[]]);
        assert_eq!(g.groups(), [vec![2], vec![1], vec![0]]);
        assert!(g.groups().iter().all(|grp| !g.is_cycle(grp)));
    }

    #[test]
    fn units_that_depend_on_one_another_are_one_group() {
        // str and tcon depend on each other, and both on core
        let g = graph(&["app", "str", "tcon", "core"], &[&[1, 3], &[2, 3], &[1, 3], &[]]);
        let groups = g.groups();
        assert_eq!(groups, [vec![3], vec![1, 2], vec![0]]);
        assert!(g.is_cycle(&groups[1]));
        assert!(!g.is_cycle(&groups[0]));
    }

    #[test]
    fn a_unit_depending_on_itself_is_a_cycle_of_one() {
        let g = graph(&["a"], &[&[0]]);
        assert_eq!(g.groups(), [vec![0]]);
        assert!(g.is_cycle(&[0]));
    }

    #[test]
    fn adding_and_linking_twice_changes_nothing() {
        let mut g = Graph::new();
        let a = g.add("a");
        let b = g.add("b");
        assert_eq!(g.add("a"), a);
        g.link(a, b);
        g.link(a, b);
        assert_eq!(g.deps(a), [b]);
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn the_drawing_shows_each_unit_below_what_depends_on_it() {
        let mut g = graph(&["app", "str", "tcon", "core", "geo"], &[&[4, 1, 3], &[2, 3], &[1, 3], &[], &[3]]);
        g.note(1, "library");
        g.note(2, "library");
        let text = g.render(&[0]);
        let want = "\
app
+-- geo
|   `-- core
+-- str [library]
|   +-- tcon [library]
|   |   +-- str [library] (cycle)
|   |   `-- core
|   `-- core
`-- core

analysed in order: core, geo, {str, tcon}, app
";
        assert_eq!(text, want);
    }

    #[test]
    fn the_drawing_can_label_each_unit() {
        let g = graph(&["app", "core"], &[&[1], &[]]);
        let text = g.render_with(&[0], &|i| format!("<{}>", g.name(i)));
        assert_eq!(text, "<app>\n`-- <core>\n\nanalysed in order: <core>, <app>\n");
    }

    #[test]
    fn a_unit_drawn_already_is_not_drawn_again() {
        let g = graph(&["a", "b", "shared", "leaf"], &[&[2], &[2], &[3], &[]]);
        let text = g.render(&[0, 1]);
        assert!(text.starts_with("a\n`-- shared\n    `-- leaf\nb\n`-- shared (see above)\n"), "{text}");
    }
}
