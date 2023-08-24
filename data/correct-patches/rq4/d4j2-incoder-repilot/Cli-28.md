# Repilot Patch

```
                    continue;
```

# Developer Patch

```
                    continue;
```

# Context

```
--- bug/Cli-28/src/java/org/apache/commons/cli/Parser.java

+++ fix/Cli-28/src/java/org/apache/commons/cli/Parser.java

@@ -287,7 +287,7 @@

                 {
                     // if the value is not yes, true or 1 then don't add the
                     // option to the CommandLine
-                    break;
+                    continue;
                 }
 
                 cmd.addOption(opt);
```

# Note

