# Repilot Patch

```
            throw new ParseException("Unsupported value type: `" + clazz.getName() + "`");
```

# Developer Patch

```
            throw new ParseException("Unable to handle the class: " + clazz);
```

# Context

```
--- bug/Cli-40/src/main/java/org/apache/commons/cli/TypeHandler.java

+++ fix/Cli-40/src/main/java/org/apache/commons/cli/TypeHandler.java

@@ -100,7 +100,7 @@

         }
         else
         {
-            return null;
+            throw new ParseException("Unsupported value type: `" + clazz.getName() + "`");
         }
     }
```

# Note

