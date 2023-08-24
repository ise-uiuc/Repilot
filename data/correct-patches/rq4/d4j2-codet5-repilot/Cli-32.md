# Repilot Patch

```

```

# Developer Patch

```

```

# Context

```
--- bug/Cli-32/src/main/java/org/apache/commons/cli/HelpFormatter.java

+++ fix/Cli-32/src/main/java/org/apache/commons/cli/HelpFormatter.java

@@ -934,11 +934,8 @@

         
         // if we didn't find one, simply chop at startPos+width
         pos = startPos + width;
-        while ((pos <= text.length()) && ((c = text.charAt(pos)) != ' ')
-               && (c != '\n') && (c != '\r'))
-        {
-            ++pos;
-        }        
+
+
         return pos == text.length() ? -1 : pos;
     }
```

# Note

