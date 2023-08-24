# Repilot Patch

```

            pos = findWrapPos(text, width, 0);
```

# Developer Patch

```
            pos = findWrapPos(text, width, 0);
```

# Context

```
--- bug/Cli-8/src/java/org/apache/commons/cli/HelpFormatter.java

+++ fix/Cli-8/src/java/org/apache/commons/cli/HelpFormatter.java

@@ -809,7 +809,8 @@

         while (true)
         {
             text = padding + text.substring(pos).trim();
-            pos = findWrapPos(text, width, nextLineTabStop);
+
+            pos = findWrapPos(text, width, 0);
 
             if (pos == -1)
             {
```

# Note

