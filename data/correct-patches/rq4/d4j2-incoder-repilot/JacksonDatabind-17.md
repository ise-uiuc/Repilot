# Repilot Patch

```
                                && !TreeNode.class.isAssignableFrom(t.getRawClass()));
```

# Developer Patch

```
                                && !TreeNode.class.isAssignableFrom(t.getRawClass()));
```

# Context

```
--- bug/JacksonDatabind-17/src/main/java/com/fasterxml/jackson/databind/ObjectMapper.java

+++ fix/JacksonDatabind-17/src/main/java/com/fasterxml/jackson/databind/ObjectMapper.java

@@ -177,7 +177,7 @@

                 return (t.getRawClass() == Object.class)
                         || (!t.isConcrete()
                                 // [databind#88] Should not apply to JSON tree models:
-                        || TreeNode.class.isAssignableFrom(t.getRawClass()));
+                                && !TreeNode.class.isAssignableFrom(t.getRawClass()));
 
             case NON_FINAL:
                 while (t.isArrayType()) {
```

# Note

