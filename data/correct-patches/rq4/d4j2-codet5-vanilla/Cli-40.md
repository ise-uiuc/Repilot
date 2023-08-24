# Repilot Patch

```

            throw new ParseException("Unknown object type : " + clazz);
```

# Developer Patch

```
            throw new ParseException("Unable to handle the class: " + clazz);
```

# Context

```
--- bug/Cli-40/src/main/java/org/apache/commons/cli/TypeHandler.java

+++ fix/Cli-40/src/main/java/org/apache/commons/cli/TypeHandler.java

@@ -100,7 +100,8 @@

         }
         else
         {
-            return null;
+
+            throw new ParseException("Unknown object type : " + clazz);
         }
     }
```

# Note

