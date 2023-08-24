# Repilot Patch

```
nextLineTabStop = 1;
```

# Developer Patch

```
            nextLineTabStop = 1;
```

# Context

```
--- bug/Cli-25/src/java/org/apache/commons/cli/HelpFormatter.java

+++ fix/Cli-25/src/java/org/apache/commons/cli/HelpFormatter.java

@@ -822,7 +822,7 @@

         if (nextLineTabStop >= width)
         {
             // stops infinite loop happening
-            nextLineTabStop = width - 1;
+nextLineTabStop = 1;
         }
 
         // all following lines must be padded with nextLineTabStop space
```

# Note

