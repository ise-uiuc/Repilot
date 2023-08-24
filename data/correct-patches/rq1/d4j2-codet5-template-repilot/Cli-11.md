# Repilot Patch

```
if (option.hasArg() && option.hasArgName())
```

# Developer Patch

```
        if (option.hasArg() && option.hasArgName())
```

# Context

```
--- bug/Cli-11/src/java/org/apache/commons/cli/HelpFormatter.java

+++ fix/Cli-11/src/java/org/apache/commons/cli/HelpFormatter.java

@@ -629,7 +629,7 @@

         }
 
         // if the Option has a value
-        if (option.hasArg() && (option.getArgName() != null))
+if (option.hasArg() && option.hasArgName())
         {
             buff.append(" <").append(option.getArgName()).append(">");
         }
```

# Note

