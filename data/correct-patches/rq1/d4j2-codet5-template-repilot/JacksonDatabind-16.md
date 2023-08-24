# Repilot Patch

```

        return previous == null
                || !previous.equals(ann);
```

# Developer Patch

```
        return (previous == null) || !previous.equals(ann);
```

# Context

```
--- bug/JacksonDatabind-16/src/main/java/com/fasterxml/jackson/databind/introspect/AnnotationMap.java

+++ fix/JacksonDatabind-16/src/main/java/com/fasterxml/jackson/databind/introspect/AnnotationMap.java

@@ -109,7 +109,9 @@

             _annotations = new HashMap<Class<? extends Annotation>,Annotation>();
         }
         Annotation previous = _annotations.put(ann.annotationType(), ann);
-        return (previous != null) && previous.equals(ann);
+
+        return previous == null
+                || !previous.equals(ann);
     }
 }
```

# Note

