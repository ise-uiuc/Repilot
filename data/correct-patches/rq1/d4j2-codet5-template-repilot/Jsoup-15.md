# Repilot Patch

```
} else if (StringUtil.in(name, "base", "basefont", "bgsound", "command", "link", "meta", "noframes", "style", "title", "script")) {
```

# Developer Patch

```
                    } else if (StringUtil.in(name, "base", "basefont", "bgsound", "command", "link", "meta", "noframes", "script", "style", "title")) {
```

# Context

```
--- bug/Jsoup-15/src/main/java/org/jsoup/parser/TreeBuilderState.java

+++ fix/Jsoup-15/src/main/java/org/jsoup/parser/TreeBuilderState.java

@@ -280,7 +280,7 @@

                             if (!html.hasAttr(attribute.getKey()))
                                 html.attributes().put(attribute);
                         }
-                    } else if (StringUtil.in(name, "base", "basefont", "bgsound", "command", "link", "meta", "noframes", "style", "title")) {
+} else if (StringUtil.in(name, "base", "basefont", "bgsound", "command", "link", "meta", "noframes", "style", "title", "script")) {
                         return tb.process(t, InHead);
                     } else if (name.equals("body")) {
                         tb.error(this);
```

# Note

