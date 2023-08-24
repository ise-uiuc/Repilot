# Repilot Patch

```

            } else if (("td".equals(name) || "th".equals(name) && !last)) {
```

# Developer Patch

```
            } else if (("td".equals(name) || "th".equals(name) && !last)) {
```

# Context

```
--- bug/Jsoup-45/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java

+++ fix/Jsoup-45/src/main/java/org/jsoup/parser/HtmlTreeBuilder.java

@@ -391,7 +391,8 @@

             if ("select".equals(name)) {
                 transition(HtmlTreeBuilderState.InSelect);
                 break; // frag
-            } else if (("td".equals(name) || "td".equals(name) && !last)) {
+
+            } else if (("td".equals(name) || "th".equals(name) && !last)) {
                 transition(HtmlTreeBuilderState.InCell);
                 break;
             } else if ("tr".equals(name)) {
```

# Note

