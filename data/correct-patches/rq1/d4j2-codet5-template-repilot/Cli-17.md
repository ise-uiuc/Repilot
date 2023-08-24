# Repilot Patch

```

                break;
```

# Developer Patch

```
                break;
```

# Context

```
--- bug/Cli-17/src/java/org/apache/commons/cli/PosixParser.java

+++ fix/Cli-17/src/java/org/apache/commons/cli/PosixParser.java

@@ -300,6 +300,8 @@

             else if (stopAtNonOption)
             {
                 process(token.substring(i));
+
+                break;
             }
             else
             {
```

# Note

