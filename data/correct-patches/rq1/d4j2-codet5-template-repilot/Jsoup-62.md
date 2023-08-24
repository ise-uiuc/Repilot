# Repilot Patch

```
String name = t.asEndTag().name();
```

# Developer Patch

```
            String name = t.asEndTag().name(); // matches with case sensitivity if enabled
```

# Context

```
--- bug/Jsoup-62/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java

+++ fix/Jsoup-62/src/main/java/org/jsoup/parser/HtmlTreeBuilderState.java

@@ -761,7 +761,7 @@

         }
 
         boolean anyOtherEndTag(Token t, HtmlTreeBuilder tb) {
-            String name = t.asEndTag().normalName();
+String name = t.asEndTag().name();
             ArrayList<Element> stack = tb.getStack();
             for (int pos = stack.size() -1; pos >= 0; pos--) {
                 Element node = stack.get(pos);
```

# Note

