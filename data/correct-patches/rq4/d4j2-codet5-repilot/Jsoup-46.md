# Repilot Patch

```

                            accum.append("&#xa0;");
```

# Developer Patch

```
                            accum.append("&#xa0;");
```

# Context

```
--- bug/Jsoup-46/src/main/java/org/jsoup/nodes/Entities.java

+++ fix/Jsoup-46/src/main/java/org/jsoup/nodes/Entities.java

@@ -115,7 +115,8 @@

                         if (escapeMode != EscapeMode.xhtml)
                             accum.append("&nbsp;");
                         else
-                            accum.append(c);
+
+                            accum.append("&#xa0;");
                         break;
                     case '<':
                         if (!inAttribute)
```

# Note

